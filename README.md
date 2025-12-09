python3 - << 'EOF'
import onnx
model = onnx.load("your_model.onnx")
print("INPUTS:")
for i in model.graph.input:
    print("  ", i.name)
print("OUTPUTS:")
for o in model.graph.output:
    print("  ", o.name)
EOF

