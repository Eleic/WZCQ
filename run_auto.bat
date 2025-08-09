@echo off
echo Starting WZCQ Automated Workflow
python auto_workflow.py --all --data-collection-duration 3600 --training-epochs 3
pause