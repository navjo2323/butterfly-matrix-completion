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

exe="run_butterfly_completion.py"                               # ← your original file
ts=$(date '+%Y%m%d_%H%M%S')                  # e.g. 20250518_2147 23

# split name and extension so the stamp goes before the dot
base=${exe%.*}                               # "file"
ext=${exe##*.}                               # "txt"  (empty if no dot)

pyfile="${base}_${ts}.${ext}"
cp -- "$rootdir/$exe" $pyfile 


# ── 1. declare the values you want ────────────────────────────────────────────
alg='ALS'
regu=1e-10
get_true_rank=0
lowrank_only=0
r_BF=11 # defining BF ranks
r_LR=11 # defining rank for the initial low-rank completion
# nnz_bf='6*r_BF*I*np.log2(I)'
nnz_bf='6488064'

# lowrank_only=1
# r_LR=60
# nnz_bf='10*r_LR*I'

tol=1e-3
L=10
c=4
kernel=1 # 1: Green's function 2: 2D Radon transform 3: 1D Radon transform
real=1 # 1: real-valued kernels, 0: complex-valued kernels

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
update_py_var regu   "$regu"
if [[ $lowrank_only == 0 ]]; then
    update_py_var r_BF   "$r_BF"
fi
update_py_var nnz_bf   "$nnz_bf"
update_py_var r_LR   "$r_LR"
update_py_var lowrank_only   "$lowrank_only"
update_py_var get_true_rank   "$get_true_rank"
update_py_var tol "$tol"
update_py_var L      "$L"
update_py_var c "$c"
update_py_var kernel        "$kernel"
update_py_var real        "$real"

logname=a.out_L${L}_c${c}_lowrank_only${lowrank_only}_rBF${r_BF}_nnz${nnz_bf}_rLR${r_LR}_regu${regu}_alg${alg}_tol${tol}_kernel${kernel}_real${real}

python -u ${pyfile} | tee ${logname}_${ts}
rm ${pyfile}
