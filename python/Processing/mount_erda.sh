# file: mount_erda.sh
#!/bin/bash
key=.ssh/id_rsa
user=saravm@sodas.ku.dk
erdadir=data
mnt=data
if [ -f "$key" ]
then
    mkdir -p ${mnt}
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${key} 
else
    echo "'${key}' is not an ssh key"
fi

