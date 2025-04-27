with open('data.ply') as f:
    lines = f.readlines()

# Find header end
header_end = 0
for i, line in enumerate(lines):
    if line.strip() == 'end_header':
        header_end = i
        break

# Get vertex and face counts from header
vertex_count = None
face_count = None
for line in lines[:header_end]:
    if line.startswith('element vertex'):
        vertex_count = int(line.split()[-1])
    if line.startswith('element face'):
        face_count = int(line.split()[-1])

# Count actual data lines
actual_vertex_lines = lines[header_end+1:header_end+1+vertex_count]
actual_face_lines = lines[header_end+1+vertex_count:header_end+1+vertex_count+face_count]

print(f"Header vertex count: {vertex_count}, actual: {len(actual_vertex_lines)}")
print(f"Header face count: {face_count}, actual: {len(actual_face_lines)}")