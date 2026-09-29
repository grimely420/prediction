import socket
import select
import sys

def forward(source, destination):
    string = ' '
    while string:
        string = source.recv(4096)
        if string:
            destination.sendall(string)
        else:
            source.shutdown(socket.SHUT_RD)
            destination.shutdown(socket.SHUT_WR)
            break

def main():
    local_host = '127.0.0.1'
    local_port = 5000
    remote_host = '192.168.0.58'
    remote_port = 5000

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((local_host, local_port))
    server.listen(50)
    print(f"Proxy listening on {local_host}:{local_port} -> {remote_host}:{remote_port}", flush=True)

    inputs = [server]
    pairs = {}

    while True:
        readable, _, exceptional = select.select(inputs, [], inputs, 1.0)
        for s in readable:
            if s is server:
                client_sock, _ = server.accept()
                try:
                    remote_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    remote_sock.connect((remote_host, remote_port))
                    inputs.extend([client_sock, remote_sock])
                    pairs[client_sock] = remote_sock
                    pairs[remote_sock] = client_sock
                except Exception as e:
                    client_sock.close()
            else:
                try:
                    data = s.recv(8192)
                    if data:
                        pairs[s].sendall(data)
                    else:
                        partner = pairs.pop(s, None)
                        if partner:
                            pairs.pop(partner, None)
                            if partner in inputs:
                                inputs.remove(partner)
                            partner.close()
                        if s in inputs:
                            inputs.remove(s)
                        s.close()
                except Exception:
                    partner = pairs.pop(s, None)
                    if partner:
                        pairs.pop(partner, None)
                        if partner in inputs:
                            inputs.remove(partner)
                        partner.close()
                    if s in inputs:
                        inputs.remove(s)
                    s.close()

if __name__ == '__main__':
    main()
