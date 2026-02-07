#!/bin/bash -l

#SBATCH --account=m2957
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --constraint=cpu
#SBATCH -t 47:59:00
#SBATCH -J QTT_completion
#SBATCH --mail-user=liuyangzhuan@lbl.gov


module load python
rootdir=$PWD/../
export PYTHONPATH=$rootdir:$PYTHONPATH
outputdir=output

cd $rootdir
mkdir -p $outputdir
cd $outputdir

exe="run_qtt_completion.py"                               # ← your original file
ts=$(date '+%Y%m%d_%H%M%S')                  # e.g. 20250518_2147 23

# split name and extension so the stamp goes before the dot
base=${exe%.*}                               # "file"
ext=${exe##*.}                               # "txt"  (empty if no dot)

pyfile="${base}_${ts}.${ext}"
cp -- "$rootdir/$exe" $pyfile 


# ── 1. declare the values you want ────────────────────────────────────────────
v2=0
alg='ADF'
regu=0
r_LR=13
start=100 # defining QTT ranks
nnz_qtt='16*start**2*np.log2(I)'
tol=1e-3
L=12
c=4
kernel=1 # 1: Green's function 2: 2D Radon transform 3: 1D Radon transform
real=1 # 1: real-valued kernels, 0: complex-valued kernels
num_iters=20
get_true_rank=0
# ─────────────────────────────────────────────────────────────────────────────

# helper: update/insert one line  var = value   (comment preserved)
update_py_var() {
    local var="$1" newval="$2"
    local quote=""
    # Only quote true strings, NOT expressions
    if [[ $newval =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        quote="'"
    elif [[ $newval =~ ^\".*\"$ || $newval =~ ^\'.*\'$ ]]; then
        quote=""
    else
        quote=""
    fi
    # 1st - try to replace the line if it exists (ignore trailing comment)
    # 2nd - if not present, append a new assignment
    if grep -qE "^[[:space:]]*$var[[:space:]]*=" "$pyfile"; then
        sed -Ei "s|^([[:space:]]*$var[[:space:]]*=[[:space:]]*).*|\1${quote}${newval}${quote}|g" "$pyfile"
    else
        echo -e "\n$var = ${quote}${newval}${quote}" >> "$pyfile"
    fi
}

# --- 2. overwrite the Python file -------------------------------------------
update_py_var alg   "$alg"
update_py_var v2   "$v2"
update_py_var regu   "$regu"
update_py_var start   "$start"
update_py_var nnz_qtt   "$nnz_qtt"
update_py_var r_LR   "$r_LR"
update_py_var tol "$tol"
update_py_var L      "$L"
update_py_var c "$c"
update_py_var kernel        "$kernel"
update_py_var num_iters        "$num_iters"
update_py_var get_true_rank        "$get_true_rank"
update_py_var real        "$real"

logname=a.out_L${L}_c${c}_rTT${start}_nnz${nnz_qtt}_rLR${r_LR}_regu${regu}_alg${alg}_v2${v2}_tol${tol}_kernel${kernel}_real${real}

python -u ${pyfile} | tee ${logname}_${ts}
rm ${pyfile}