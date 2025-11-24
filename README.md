# Director explanation
The code is devided in to different parts:

```text
📂 LDMGI_code_toRun/
├── 📂 data/
│   ├── 📄 jaffe.npy
│   └── ...
├── 📂 LDMGI/                 # The model architecture
├── 🐍 download_Coil.py       # Data loader script
├── 🐍 load_jaffe.py          # Data loader script
├── 🐍 load_xxxx.py           # Data loader script of other test sets
└── 🚀 main.py                # Main execution script
```
## Test Instruction:

Before running the code, several parameters should be changed manually in main.py:

line 33: change the number of class c

line 42~49: comment or uncomment the line to chose which data set is being tested

line 51: set the Name in the output

line 66: change the number of trials. The defult is set to one. Our testing results use n_trails = 20 (as the paper specifies)


The data above are modified refering to this table:

<img width="416" height="254" alt="image" src="https://github.com/user-attachments/assets/6ace340e-93f6-4c36-8c2e-0b00c9accec9" />

After change the corresonding parameters, you can press 'Run' button and test the code.

