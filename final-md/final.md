 echo $RIG
--calib hw_calib_teleop.npz --base-frame base_frame_teleop.npz --puck-profile puck_profile.json --ee-height-npz ee_height.npz --camera-index 0 --cam-fps 30 --cam-auto-exposure 1 --cam-exposure 150 --cam-auto-priority 0
echo $EKF
--ekf-params mini_table_ekf.npz --sigma-px 0.56 --max-puck-speed 30.0
echo $MOTION
--flat-mallet --rt-home --rt-home-time 5 --max-joint-vel 1.8,1.4,1.4,1.4,1.4,1.4 --max-joint-accel 70,140,65,130,40,40

**run** 
python -u -m mini_platform.hw.run_hw_match   --stage play   --log clear_early.npz   --calib hw_calib_teleop.npz   --base-frame base_frame_teleop.npz   --puck-profile puck_profile.json   --camera-index 0   --cam-fps 30   --cam-exposure 150   --cam-auto-exposure 1   --cam-auto-priority 0   --ekf-params mini_table_ekf.npz   --sigma-px 0.56   --max-puck-speed 10.0   --lead-time 0.05   --flat-mallet   --ee-height-npz ee_height.npz   --max-joint-vel 2.2,1.8,1.8,1.8,1.8,1.8   --max-joint-accel 90,160,85,135,40,40   --rt-home   --rt-home-time 2   --goalie   --defend-width-frac 1.2   --max-plan-steps 3   --ee-height 0.1230   --gripper-mm 22   --min-goal-dist 0.002   --j1-limit -0.7295 0.7186 --planner vpsto

**kill can**

sudo pkill -x slcand

**tune puck**

python -m mini_platform.hw.tune_puck --camera-index $CAM --resolution 640x480 \
    --cam-auto-exposure 1 --cam-exposure 150 --cam-auto-priority 0 \
    --out puck_profile.json