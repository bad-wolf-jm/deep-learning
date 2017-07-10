import paramiko
import sys
import time


client = paramiko.client.SSHClient()
client.load_system_host_keys()
client.connect('192.168.0.111', username='jihemme', password='marijke')
#shell = client.invoke_shell()
stdin, stdout, stderr = client.exec_command('cd /Users/jihemme/Python/DJ/deep-learning/deeplearn/twitter_sentiment/ && /usr/local/bin/python3 -u -m models.byte_cnn.train')
#stdin, stdout, stderr = shell.exec_command('ls -l')
#print stderr.read()
#print stdin
#print stdout
#print stderr
#for i in range (100):
#print 'comment'
#stdout.channel.recv_exit_status()
#print 'process done'
line = stdout.readline()
#print line
while line != '':#
    sys.stdout.write('[entered]  '+line)
#    #stdout.write("HELLO")
#    sys.stdout.write("foo"+line)
#    #print 'ddd'
#    #stdin.channel.sendall(line)
#    #stdin.flush()
#    #print stderr.readline(timeout = 0.5)
    line = stdout.readline()
#    #print line
#time.sleep(20)
client.close()
#print stderr.read()
#print stdout.read()
