@echo off
REM SUPPLEMENTARY held-out run with Neigh (review round 2). NOT PRE-REGISTERED; declared in SUPP_PROTOCOL.md.
REM It reads the finished pre-registered results and never changes them.
REM Double-click this file. It keeps its own window open; do not close it until it says DONE (about 1-1.5 h).
REM Re-running is safe: finished graphs are skipped; failed graphs are retried.

setlocal
set ROOT=D:\Work\Triangle Ordering
set ART=%ROOT%\audit_workspace\DOVE_AISTATS_2027_code_and_data
set PCEVAL=%ROOT%\pc_eval
set POSTHOC=%ROOT%\pc_eval_posthoc
set PKG=%ROOT%\pc_eval_neigh_supp
set DATA=%ROOT%\pc_eval_data
set RES=%ROOT%\pc_eval_results
set OUT=%ROOT%\pc_eval_neigh_supp_results
set LOG=%ROOT%\pc_eval_neigh_supp_console.log
set VOLT=%PKG%\volt\volt.exe
set NUMBA_CACHE_DIR=%PKG%\.numba_cache
set PYTHONDONTWRITEBYTECODE=1

cd /d "%ROOT%" || goto :fail
call "%ROOT%\.venv\Scripts\activate.bat" || goto :fail
cd /d "%PKG%" || goto :fail

echo [%date% %time%] Step 0/3: checking packages and VOLT
python -c "import numpy, scipy, numba; print(numpy.__version__, scipy.__version__, numba.__version__)" || goto :fail
if not exist "%VOLT%" goto :voltfail
"%VOLT%" --help > "%OUT%_volt_help.txt" 2>&1 || goto :voltfail
findstr /C:"--rank-output" "%OUT%_volt_help.txt" >nul || goto :voltfail

echo [%date% %time%] Step 1/3: smoke test on Ego-0 and CollegeMsg (1-2 min), written to %OUT%_smoke
REM The smoke run analyses two graphs only, so its table step reports the others as "not run" and exits exactly 2.
REM Any other exit code, or a missing PASSED line, is a smoke failure.
python neigh_supp.py --pc-eval "%PCEVAL%" --posthoc "%POSTHOC%" --artifact "%ART%" --results "%RES%" --data "%DATA%" --out "%OUT%_smoke" --volt "%VOLT%" --only Ego-0 CollegeMsg >> "%LOG%" 2>&1
if errorlevel 3 goto :fail
if not errorlevel 2 goto :fail
if not errorlevel 0 goto :fail
findstr /C:"N CollegeMsg: supplementary verification PASSED" "%OUT%_smoke\supp_log.txt" >nul || goto :fail
findstr /C:"V Ego-0: stored Neigh seeds" "%OUT%_smoke\supp_log.txt" >nul || goto :fail
echo [%date% %time%] Smoke test passed.

echo [%date% %time%] Step 2/3: full run (Part V: 32 suite graphs, Part N: 9 held-out graphs). Progress: %OUT%\supp_log.txt
python neigh_supp.py --pc-eval "%PCEVAL%" --posthoc "%POSTHOC%" --artifact "%ART%" --results "%RES%" --data "%DATA%" --out "%OUT%" --volt "%VOLT%" >> "%LOG%" 2>&1
if not errorlevel 0 goto :fail
if errorlevel 3 goto :fail
if errorlevel 2 goto :partial
if errorlevel 1 goto :fail

echo [%date% %time%] Step 3/3: rewriting the tables (also done at the end of step 2)
python make_supp_tables.py --results "%RES%" --out "%OUT%" >> "%LOG%" 2>&1
if not errorlevel 0 goto :fail
if errorlevel 3 goto :fail
if errorlevel 2 goto :partial
if errorlevel 1 goto :fail

echo.
echo [%date% %time%] DONE. Results: %OUT%   Console log: %LOG%
echo Tell Claude the Neigh run finished.
pause
exit /b 0

:voltfail
echo.
echo [%date% %time%] VOLT did not start. See %OUT%_volt_help.txt. If Windows blocked volt.exe, right-click it,
echo choose Properties, tick "Unblock", and run this file again. Tell Claude if it still fails.
pause
exit /b 1

:partial
echo.
echo [%date% %time%] FINISHED WITH FAILURES on at least one graph (see %OUT%\supp_log.txt, line "SUPPLEMENTARY FAILURES").
echo Re-running retries failed graphs. Tell Claude.
pause
exit /b 2

:fail
echo.
echo [%date% %time%] STOPPED WITH AN ERROR. See %LOG% (and %OUT%\supp_log.txt). Re-running this file resumes.
pause
exit /b 1
