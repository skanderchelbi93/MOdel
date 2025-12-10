```bash
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
    launch_fragments:=foundationpose \
    interface_specs_file:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/quickstart_interface_specs.json \
    mesh_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/Drill/drill.obj \
    texture_path:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/Mustard/texture_map.png \
    score_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
    refine_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
    model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/best.onnx \
    engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/best.plan
`````

```bash
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
  launch_fragments:=foundationpose \
  interface_specs_file:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/quickstart_interface_specs.json \
  mesh_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/Drill/drill.obj \
  texture_path:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/Mustard/texture_map.png \
  score_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
  refine_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
  model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/best.onnx \
  engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/best.plan \
  input_tensor_names:='["images"]' \
  input_binding_names:='["images"]' \
  output_tensor_names:='["output0"]' \
  output_binding_names:='["output0"]'

`````
