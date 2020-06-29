# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$DIR_TMP/src
export PYTHONPATH=$PYTHONPATH:$DIR_TMP/utest

echo PYTHONPATH=$PYTHONPATH