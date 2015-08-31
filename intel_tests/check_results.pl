#!/usr/bin/env perl

use strict;
use Data::Dumper;

my $outcomes;

$outcomes->{"0_skipped"}  = "(MPITEST_SKIP|MPITEST skip)";
$outcomes->{"1_passed"} = "MPITEST_results:.*PASSED";
$outcomes->{"2_failed"} = "MPITEST_results:.*FAILED";
$outcomes->{"3_broken_pipe"} = "Broken pipe";
$outcomes->{"4_failed_assert"} = "Assertion.*failed";
$outcomes->{"5_signaled"} = "Signal:[0-9]";
$outcomes->{"6_killed"} = "Killed";
$outcomes->{"7_aborted"} = "[0-9]+ additional processes aborted";

sub check_file {
    my ($filename) = shift;

    print "Checking results: $filename\n";
    open FILE, $filename || die("Cannot open file: $filename");
    my @contents;
    my $save = 0;
    my $in_test = 0;
    my $results;
    my $total_count = 0;
    my $test_name;

    # Reset the counters
    foreach my $key (keys(%$outcomes)) {
        $results->{$key}->{count} = 0;
    }
    $results->{unknown}->{count} = 0;

    # Read the file
    while (<FILE>) {
	chomp;
	# Get only the results -- skip the header and footer
	if (/\[\[\[ START OF TESTS \]\]\]/) {
	    $save = 1;
	    next;
	} elsif (/\[\[\[ END OF TESTS \]\]\]/) {
	    last;
	}

	if ($save) {
            if ($in_test) {
                if (/^---- MPI_/) {
                    $in_test = 0;

                    my $found = 0;
                    my $out_key;
                    my $out_line;
                    for (my $i = 0; $i <= $#contents; ++$i) {
                        foreach my $key (sort(keys(%$outcomes))) {
                            my $str = "\$found = (\$contents[\$i] =~ /$outcomes->{$key}/);";
                            eval $str;
                            if ($found) {
                                $out_key = $key;
                                $out_line = $contents[$i];
                                last;
                            }
                        }
                        last if ($found);
                    }
                    # Now save the results
                    $out_key = "unknown"
                        if (!$found);

                    ++$results->{$out_key}->{count};
                    $results->{$out_key}->{tests}->{$test_name}->{output} = 
                        @contents;
                    $results->{$out_key}->{match}->{$test_name}->{match} =
                        $out_line;

                    # Reset for the next test
                    @contents = ();
                }

                # Nope, this was just another line in the text
                else {
                    push(@contents, $_);
                }
            }
            elsif (/^\+\+\+\+ (MPI_.+)/) {
                # Starting a new test
                $test_name = $1;
                $in_test = 1;
                ++$total_count;
            }
	}
    }

    print "Found $total_count total tests\n";
    # Reset the counters
    foreach my $key (sort(keys(%$outcomes))) {
        print "$key: $results->{$key}->{count}\n";
        if ($key ne "1_passed" && $key ne "0_skipped") {
            foreach my $t (sort(keys(%{$results->{$key}->{tests}}))) {
                print "==> $t\n";
                if ($results->{$key}->{$t}->{match}) {
                    print "$results->{$key}->{$t}->{match}\n";
                }
            }
        }
    }
    print "unknown: $results->{unknown}->{count}\n";
    foreach my $t (keys(%{$results->{unknown}->{tests}})) {
        print "==> $t\n";
    }
}

if ($#ARGV >= 0) {
    foreach my $file (@ARGV) {
	check_file($file);
    }
} else {
    opendir(DIR, ".") || die("Cannot open directory to find results files");
    my @files = grep { /\.out$/ && -f "./$_" } readdir(DIR);
    close(DIR);
    my $checked = 0;
    foreach my $file (@files) {
        print "---------------------------------------------------------------------------\n";
	check_file($file);
        $checked = 1;
    }
    print "---------------------------------------------------------------------------\n"
        if ($checked);
}

exit(0);
