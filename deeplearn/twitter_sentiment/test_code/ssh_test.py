from paramiko.client import SSHClient

client = SSHClient()
client.load_system_host_keys()
client.connect('10.137.11.91', username='jalbert', password='gameloft2017')
channel = client.invoke_shell()

stdin, stdout, stderr = channel.exec_command('workon buzzometer-dev.gameloft.org & python --version')
print(stdout.readlines())
stdin, stdout, stderr = client.exec_command('python --version')
print(stdout.readlines())
print(stderr.readlines())
client.close()
