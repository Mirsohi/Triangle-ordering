@echo off
REM One-click runner for the pre-registered evaluation (smoke test, full run, tables).
REM Double-click this file. It keeps its own window open; do not close it until it says DONE.
REM Re-running is safe: finished graphs are skipped.

setlocal
set ROOT=D:\Work\Triangle Ordering
set ART=%ROOT%\audit_workspace\DOVE_AISTATS_2027_code_and_data
set DATA=%ROOT%\pc_eval_data
set OUT=%ROOT%\pc_eval_results
set PAPER=%ROOT%\paper_revision_20260928
set LOG=%ROOT%\pc_eval_results_console.log

cd /d "%ROOT%" || goto :fail
call "%ROOT%\.venv\Scripts\activate.bat" || goto :fail
cd /d "%ROOT%\pc_eval" || goto :fail

echo [%date% %time%] Checking packages...
python -c "import numpy, scipy, numba, networkx; print(numpy.__version__, scipy.__version__, numba.__version__, networkx.__version__)" || goto :fail

echo [%date% %time%] Step 1/3: smoke test (2-4 min)
python run_eval.py --artifact "%ART%" --data "%DATA%" --out "%OUT%" --part both --quick >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] Step 2/3: full run, 32-graph suite + 10 held-out graphs (about 1-2 h). Progress: %OUT%\run_log.txt
python run_eval.py --artifact "%ART%" --data "%DATA%" --out "%OUT%" --part both >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] Step 3/3: writing the tables into the paper folder
python make_tables.py --out "%OUT%" --paper "%PAPER%" >> "%LOG%" 2>&1 || goto :fail

echo.
echo [%date% %time%] DONE. Results: %OUT%   Console log: %LOG%
echo Tell Claude the run finished.
pause
exit /b 0

:fail
echo.
echo [%date% %time%] STOPPED WITH AN ERROR. See %LOG% (and %OUT%\run_log.txt). Re-running this file resumes.
pause
exit /b 1
