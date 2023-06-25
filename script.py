import subprocess

print("-"*50)
print("Running script...")
print("Running first arguments")

# First invocation of main.py
args1 = [
    'python3',
    'main.py',
    '--arg0', 'Havering',
    '--arg1', 'M25/5790A',
    '--arg2', '2018',
    '--arg3', '1.5',
    '--arg4', '11:5/2',
    '--arg5', '-1',
    '--arg6', '/home/ah2719/FYP/havering_m25_5790a.tif'
]

subprocess.run(args1)

# Second invocation of main.py with different arguments
args2 = [
    'python3',
    'main.py',
    '--arg0', 'Trafford',
    '--arg1', 'M60/9086b',
    '--arg2', '2018',
    '--arg3', '2.5',
    '--arg4', '11:6/2',
    '--arg5', '65',
    '--arg6', '/home/ah2719/FYP/trafford_m60_9086b.tif'
]

print("-"*50)
print("Running second arguments")
subprocess.run(args2)

print("-"*50)

print("Sucessfully finished running script!")
print("-"*50)