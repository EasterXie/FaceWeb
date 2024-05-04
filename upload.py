import os
import paramiko

def upload_folder_to_server(local_folder, remote_folder, server_ip, username, password):
    # 创建一个SSH客户端
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 连接到服务器
    client.connect(server_ip, username=username, password=password)

    # 创建一个SFTP客户端
    sftp = client.open_sftp()

    # 遍历本地文件夹
     # 检查远程路径是否存在，如果不存在则创建一个新的文件夹
    
    for foldername, subfolders, filenames in os.walk(local_folder):
        try:
            sftp.stat(remote_folder)
        except IOError:
            sftp.mkdir(remote_folder)
        for filename in filenames:
            # 构造本地文件的完整路径
            local_path = f"{foldername}/{filename}"
            print(local_path)
            # 构造远程文件的完整路径
            remote_path = os.path.join(remote_folder, foldername[len(local_folder):], filename)
            remote_path2 = f"{remote_folder}/{filename}"
            print(remote_path2)
            # 上传文件到服务器
            
            sftp.put(local_path, remote_path2)
           
    # 关闭连接
    client.close()

