__conda_setup="$(CONDA_REPORT_ERRORS=false '/nfs/pic.es/user/s/sgonzalez/anaconda2/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/nfs/pic.es/user/s/sgonzalez/anaconda2/etc/profile.d/conda.sh" ]; then
        . "/nfs/pic.es/user/s/sgonzalez/anaconda2/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/nfs/pic.es/user/s/sgonzalez/anaconda2/bin:$PATH"
    fi
fi
unset __conda_setup

