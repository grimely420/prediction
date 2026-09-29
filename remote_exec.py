import sys
import paramiko

def run_remote(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.0.58', username='chain-deaction', password='Homiez@420', timeout=15)
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    exit_status = stdout.channel.recv_exit_status()
    ssh.close()
    return exit_status, out, err

if __name__ == '__main__':
    command = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "ls -la"
    code, out, err = run_remote(command)
    if out:
        sys.stdout.buffer.write(out.encode('utf-8', errors='replace'))
    if err:
        sys.stderr.buffer.write(err.encode('utf-8', errors='replace'))
    sys.exit(code)
