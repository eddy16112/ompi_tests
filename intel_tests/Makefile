.PHONY:	all clean distclean run

FILE =
SHELL = /bin/sh
MPIRUN = mpirun --bynode -np 8
#DEBUG = xterm -e gdb
#DEBUG = valgrind --tool=memcheck --num-callers=10 --db-attach=no --log-file=out --leak-check=yes --show-reachable=no
PARAMS = -- $(DEBUG) `pwd`/src/
all:
	@echo Usage: make TESTNAME

clean:
	$(RM) core*
	(cd src ; $(MAKE) clean)
	(cd src/cuda ; $(MAKE) clean)

distclean: clean
	$(RM) *.out *.log

run:
	@if test -z "$(FILE)"; then \
	  echo "Must specify a FILE on the 'make' command line"; \
	  echo "Example: 'make run FILE=foo'"; \
	elif test ! -r "$(FILE)"; then \
	  echo "The FILE $(FILE) does not seem to exist, or is unreadable"; \
	else \
	  LOG=$(FILE).$$$$.out                          ; \
	  START=`date`                                  ; \
	  if test true ; then                             \
	    echo "Intel MPI test suite"                 ; \
	    echo "Date:     `date`"                     ; \
	    echo "Host:     `uname -a`"                 ; \
	    echo "Log:      $(FILE).$$$$.out"           ; \
	    echo "Version:"                             ; \
	    ompi_info --version                         ; \
	    echo "Testfile:         $(FILE)"            ; \
	    echo "Tests:"                               ; \
	    cat $(FILE)                                 ; \
	    echo ""                                     ; \
	    echo "[[[ START OF TESTS ]]]"               ; \
	    for t in `cat $(FILE)` ; do                   \
	      echo ++++ $$t                             ; \
	      $(MAKE) $$t                               ; \
	      echo ---- $$t                             ; \
	    done                                        ; \
	    echo "[[[ END OF TESTS ]]]"                 ; \
	  fi 2>&1 | tee $$LOG                           ; \
	  echo "Tests complete. Log in $$LOG"; \
	  echo "Start: $$START"                         ; \
	  echo "End:   `date`"                          ; \
	fi

jcompile:
	@if test -z "$(FILE)"; then \
	  echo "Must specify a FILE on the 'make' command line"; \
	  echo "Example: 'make run FILE=foo'"; \
	elif test ! -r "$(FILE)"; then \
	  echo "The FILE $(FILE) does not seem to exist, or is unreadable"; \
	else \
	  START=`date`                                  ; \
	  echo "Intel MPI test suite"                   ; \
	  echo "Date:     `date`"                       ; \
	  echo "Host:     `uname -a`"                   ; \
	  echo "Log:      $(FILE).$$$$.out"             ; \
	  echo "Version:"                               ; \
	  ompi_info --version                           ; \
	  echo "Testfile:         $(FILE)"              ; \
	  echo "Tests:"                                 ; \
	  cat $(FILE)                                   ; \
	  if [ $(FILE) == "cuda_c" ]; then                \
	    testpath="src/cuda"                         ; \
          else                                            \
            testpath="src"                              ; \
          fi                                            ; \
	  echo ""                                       ; \
	  startdir=`pwd`                                ; \
	  cd `dirname $(FILE)`                          ; \
	  fileabs="`pwd`/$(FILE)"                       ; \
	  cd "$$startdir"                               ; \
	  echo "[[[ START OF COMPILE ]]]"               ; \
	  (cd "$$testpath"; cat "$$fileabs" | xargs -n 64 $(MAKE)) ; \
	  echo "[[[ END OF COMPILE ]]]"                 ; \
	  echo "Compile complete."                      ; \
	  echo "Start: $$START"                         ; \
	  echo "End:   `date`"                          ; \
	fi

# Note that this target is slow because it builds one at a time;
# parallel builds won't get you anything here.  Use the "jcompile" to
# get parallel building.  On my system, "make -j 32 jcompile
# FILE=all_tests_no_perf" takes 1.6 seconds.  The "compile" target for
# the same FILE takes 52 seconds.  You decide which is better. :-)
compile:
	@if test -z "$(FILE)"; then \
	  echo "Must specify a FILE on the 'make' command line"; \
	  echo "Example: 'make run FILE=foo'"; \
	elif test ! -r "$(FILE)"; then \
	  echo "The FILE $(FILE) does not seem to exist, or is unreadable"; \
	else \
	  START=`date`                                  ; \
	  echo "Intel MPI test suite"                   ; \
	  echo "Date:     `date`"                       ; \
	  echo "Host:     `uname -a`"                   ; \
	  echo "Log:      $(FILE).$$$$.out"             ; \
	  echo "Version:"                               ; \
	  ompi_info --version                           ; \
	  echo "Testfile:         $(FILE)"              ; \
	  echo "Tests:"                                 ; \
	  cat $(FILE)                                   ; \
	  if [ $(FILE) == "cuda_c" ]; then                \
	    testpath="src/cuda"                         ; \
          else                                            \
            testpath="src"                              ; \
          fi                                            ; \
	  echo ""                                       ; \
	  echo "[[[ START OF COMPILE ]]]"               ; \
	  for t in `cat $(FILE)` ; do                     \
	    echo ++++ $$t                               ; \
	    (cd "$$testpath" ; $(MAKE) $$t)             ; \
	    status=$$?                                  ; \
	    if test "$$status" != "0"; then               \
	      exit $$status                             ; \
	    fi                                          ; \
	    echo ---- $$t                               ; \
	  done                                          ; \
	  echo "[[[ END OF COMPILE ]]]"                 ; \
	  echo "Compile complete."                      ; \
	  echo "Start: $$START"                         ; \
	  echo "End:   `date`"                          ; \
	fi

%cuda_c:
	(cd src/cuda ; $(MAKE) $@)
	$(MPIRUN) $(PARAMS)cuda/$@

%:
	(cd src ; $(MAKE) $@)
	$(MPIRUN) $(PARAMS)$@
