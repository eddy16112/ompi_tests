#!/usr/bin/perl

#
# Driver program to run all the tests in this directory
#  iterating over the required parameters.
#
# Usage instructions:
#   - Hostfile should be in the intel_tests/ directory. The name is this file [with no path]
#     is set below.
#   - OMPI bin[ary] install directory must be prominent in your path, so it finds
#     mpirun in this directory by default
#   - Press CTRL-C to kill a stalled run, do this only once!
#   - kill the perl process to stop this wrapper, then manually clean up any leftover 
#     MPI processes
#
use strict;
# for WNOHANG
use POSIX ":sys_wait_h";

my $ignore      = "1> /dev/null 2> /dev/null";
my $output_file = "/tmp/gen-output.txt";
my $holdall     = "2>&1) > $output_file";
my $clean_out   = "mv $output_file"."-saved ../ && rm $output_file";

# Optional prefix to pass to the mpirun
my $prefix      = "";
# Number of processors to use
my $procs    = 6;
# hostfile name in directory
my $hostfile = "hostfile";
# Maximum wait time for a test before killing it, in seconds
my $max_wait = 90;

my $command;
my $prog;
my $mca;

#
# MCA PTL Params
#
my @mca_opts = ( 
#                 "tcp",
#                 "tcp,self", 
#                 "tcp,sm,self"
#                 "gm", 
#                 "gm,self", 
#                 "gm,sm,self", 
                 "sm,self"
                 );

#
# Tests to run specified in files which were provided as arguments
# 
my @progs;
my @hosts;
my @failed_tests;
my $argc = scalar(@ARGV);
my $i;
my $line;
my $file;
my $child_pid;
my $child_exitcode;

if($argc <= 0){
    print "Usage: ./run-tests.pl input_file\n";
    exit;
}
for($i = 0; $i < $argc; ++$i){
    $file = $ARGV[$i];
    if($file =~ /(-hostfile)/) {
        ++$i;
        $file = $ARGV[$i];
        $hostfile = $file;
        $output_file .= "-" . $hostfile;
        # reset these
        $holdall     = "2>&1) > $output_file";
        $clean_out   = "mv $output_file"."-saved ../ && rm $output_file";
        next;
    }
    else {
        print("Getting programs from file <".$file.">\n");
        
        if( !open(INPUT, $file) ){
            print "Error loading file: $file\n";
        next;
        }
        
        #
        # Get all programs to run from the file
        #
        while( $line = <INPUT> ){
            chop($line);
            push(@progs, $line);
        }
        close(INPUT);
    }
}

#
# If no arguments provided, we are finished
#
if(scalar(@progs) <= 0){
    print "No programs provided\n";
    exit;
}

#
# Ensure hostfile exists in this directory
#
if(!open(TMP, $hostfile)){
    print "Unable to locate hostfile <$hostfile> in the current directory\n";
    exit;
}
print "\nEnvironment\n";
print "********************\n";

$command = "mpicc --version";
print "\$ $command\n";
system($command);

$command = "mpif77 -V";
print "\$ $command\n";
if(0 != system($command) ) {
    $command = "mpif77 --version";
    system($command);
}

print "\n<Hostfile: $hostfile>\n";
print "********************\n";
while( $line = <TMP>){
    chop($line);
    print "  $line\n";
    $line =~ /\s|\r/;
    push(@hosts, $`);
}
close(TMP);

print "\n********************\n";
print "Make tests ...\n";
print "********************\n";
chdir("src");
my $rtn;
if( 0 != ($rtn = system("make clean ".join(" ", @progs))) ){
    print "\nMake failed!\n";
    exit;
}

print "\n********************\n";
print "Run All programs (".scalar(@progs)." tests. Runtime  =~ ". (($max_wait*scalar(@progs))/60) ." min)...\n";
print "********************\n";
runset(\@progs, \@mca_opts, "../$hostfile");

print "\n****Finished!****\n\n";

if(scalar(@failed_tests) > 0){
    print "\n********************\n";
    print "Failed tests...\n";
    print "********************\n";
    print "  Filename\t\tMCA Parms\n";
    for($i = 0; $i < scalar(@failed_tests); ++$i) {
        print "  ".$failed_tests[$i]."\t<".$failed_tests[++$i].">\n";
    }
}
print "\n";

system($clean_out);

exit;

#
# Iterate over all MCA PTL parameters then all programs
#
sub runset(\@\@$) {
    my ($progs, $options, $hosts) = @_;
    my $prog;
    my $mca;
    my $host;
    my $line;
    my $flag = 0;
    my $mpicommand;
    my $tm = 0;

    foreach $prog (@$progs){
        foreach $mca (@$options) {
            print "prefix:$prefix\n";
            if (length $prefix > 0) {
                $mpicommand = "mpirun --prefix $prefix -np $procs -hostfile $hosts --mca btl $mca $prog";
            } else {
                $mpicommand = "mpirun -np $procs -hostfile $hosts --mca btl $mca $prog";
            }
            print("Executing <$mpicommand>\n");

            # 
            # Fork off the child, and let the parent wait for it
            #
            if( ($child_pid = fork) ) {
                $tm = 0;
                do {
                    sleep(1);
                    waitpid($child_pid, &WNOHANG);
                    $child_exitcode = $?;
                    ++$tm;
                } while($child_exitcode < 0 && $tm <= $max_wait);
            }
            else {
                exec("(" . $mpicommand . " $holdall");
            }

            #
            # Clean up
            #  First with sig 11 to print debug info
            #  then with sig 9 to make sure we don't get zombies on remote nodes
            #
            foreach $host (@hosts) {
                $command = "ssh $host \"killall -s 11 mpirun $prog\" $ignore";
                #print("Clean up with <$command>\n");
                system($command);
            }
            foreach $host (@hosts) {
                $command = "ssh $host \"killall -s 9 mpirun $prog\" $ignore";
                #print("Deep Clean up with <$command>\n");
                system($command);
            }

            #
            # If the child didn't exit before we tried to kill it,
            # then wait for it to finish
            #
            if($child_exitcode < 0){
              waitpid($child_pid, 0);
              $child_exitcode = $?;
            }
            
            #
            # Check output file for errors
            #
            if(! open(INPUT, "<", "$output_file")) {
              print "Opps unable to open file <$output_file>. Save it for historical purposes.\n";
              $flag = 2;
            }
            else {
              while( $line = <INPUT> ){
                if( $line =~ /FAILED/) {
                  $flag = 1;
                }
              }
              close(INPUT);
            }
            
            if($child_exitcode != 0 || $flag != 0){
                push(@failed_tests, $prog);
                push(@failed_tests, $mca);

                system("echo >> $output_file"."-saved");
                system("echo Executing \"$mpicommand\" >> $output_file"."-saved");
                system("echo >> $output_file"."-saved");
                system("cat $output_file >> $output_file"."-saved");
            }
            else {
                system("echo >> $output_file"."-saved");
                system("echo Executing \"$mpicommand\" >> $output_file"."-saved");
                system("echo >> $output_file"."-saved");
            }                
            system("cat $output_file");

            $flag = 0;
        }
    }

}
