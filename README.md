python3 -c "import onnx; m=onnx.load('mymodel.onnx'); 
print('Inputs:', [i.name for i in m.graph.input]); 
print('Outputs:', [o.name for o in m.graph.output])"

