import os
import subprocess

folder = 'C:\\Users\\ASUS\\pprojects\\deep-learning-ttic\\frameworks'

for filename in os.listdir(folder):
    if filename.endswith('.tex'):
        tex_file = os.path.join(folder, filename)
        subprocess.run(['pdflatex', '-output-directory', folder, tex_file])
        
        # deletee auxiliary files
        base_filename = os.path.splitext(filename)[0]
        for ext in ['.aux', '.log', '.out']:
            aux_file = os.path.join(folder, f'{base_filename}{ext}')
            if os.path.exists(aux_file):
                os.remove(aux_file)
