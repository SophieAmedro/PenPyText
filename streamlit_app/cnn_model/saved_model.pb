#
˙ă
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%ˇŃ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02unknown8ň
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@*
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
:		@*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:@*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0

conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_2/kernel
x
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*'
_output_shapes
:@*
dtype0
o
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/bias
h
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes	
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0
Ł
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0

conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/kernel
y
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*(
_output_shapes
:*
dtype0
o
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/bias
h
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes	
:*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:*
dtype0
Ł
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:*
dtype0

conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/kernel
y
!conv_4/kernel/Read/ReadVariableOpReadVariableOpconv_4/kernel*(
_output_shapes
:*
dtype0
o
conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/bias
h
conv_4/bias/Read/ReadVariableOpReadVariableOpconv_4/bias*
_output_shapes	
:*
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:*
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:*
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:*
dtype0
Ł
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:*
dtype0

conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_5/kernel
y
!conv_5/kernel/Read/ReadVariableOpReadVariableOpconv_5/kernel*(
_output_shapes
:*
dtype0
o
conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_5/bias
h
conv_5/bias/Read/ReadVariableOpReadVariableOpconv_5/bias*
_output_shapes	
:*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0
Ł
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0

batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes	
:*
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes	
:*
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes	
:*
dtype0
Ł
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes	
:*
dtype0

classifier/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä>*"
shared_nameclassifier/kernel
y
%classifier/kernel/Read/ReadVariableOpReadVariableOpclassifier/kernel* 
_output_shapes
:
Ä>*
dtype0
w
classifier/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä>* 
shared_nameclassifier/bias
p
#classifier/bias/Read/ReadVariableOpReadVariableOpclassifier/bias*
_output_shapes	
:Ä>*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@*%
shared_nameAdam/conv_1/kernel/m

(Adam/conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_1/kernel/m*&
_output_shapes
:		@*
dtype0
|
Adam/conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_1/bias/m
u
&Adam/conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_1/bias/m*
_output_shapes
:@*
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:@*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:@*
dtype0

Adam/conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv_2/kernel/m

(Adam/conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_2/kernel/m*'
_output_shapes
:@*
dtype0
}
Adam/conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv_2/bias/m
v
&Adam/conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_2/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:*
dtype0

Adam/conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv_3/kernel/m

(Adam/conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_3/kernel/m*(
_output_shapes
:*
dtype0
}
Adam/conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv_3/bias/m
v
&Adam/conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_3/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/m

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/m

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes	
:*
dtype0

Adam/conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv_4/kernel/m

(Adam/conv_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_4/kernel/m*(
_output_shapes
:*
dtype0
}
Adam/conv_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv_4/bias/m
v
&Adam/conv_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_4/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/m

6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/m

5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:*
dtype0

Adam/conv_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv_5/kernel/m

(Adam/conv_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_5/kernel/m*(
_output_shapes
:*
dtype0
}
Adam/conv_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv_5/bias/m
v
&Adam/conv_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_5/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/m

6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/m

5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/m

6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/m

5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes	
:*
dtype0

Adam/classifier/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä>*)
shared_nameAdam/classifier/kernel/m

,Adam/classifier/kernel/m/Read/ReadVariableOpReadVariableOpAdam/classifier/kernel/m* 
_output_shapes
:
Ä>*
dtype0

Adam/classifier/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä>*'
shared_nameAdam/classifier/bias/m
~
*Adam/classifier/bias/m/Read/ReadVariableOpReadVariableOpAdam/classifier/bias/m*
_output_shapes	
:Ä>*
dtype0

Adam/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@*%
shared_nameAdam/conv_1/kernel/v

(Adam/conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_1/kernel/v*&
_output_shapes
:		@*
dtype0
|
Adam/conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_1/bias/v
u
&Adam/conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_1/bias/v*
_output_shapes
:@*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:@*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:@*
dtype0

Adam/conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv_2/kernel/v

(Adam/conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_2/kernel/v*'
_output_shapes
:@*
dtype0
}
Adam/conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv_2/bias/v
v
&Adam/conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_2/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:*
dtype0

Adam/conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv_3/kernel/v

(Adam/conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_3/kernel/v*(
_output_shapes
:*
dtype0
}
Adam/conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv_3/bias/v
v
&Adam/conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_3/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes	
:*
dtype0

Adam/conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv_4/kernel/v

(Adam/conv_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_4/kernel/v*(
_output_shapes
:*
dtype0
}
Adam/conv_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv_4/bias/v
v
&Adam/conv_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_4/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/v

6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/v

5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:*
dtype0

Adam/conv_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv_5/kernel/v

(Adam/conv_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_5/kernel/v*(
_output_shapes
:*
dtype0
}
Adam/conv_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv_5/bias/v
v
&Adam/conv_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_5/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/v

6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/v

5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/v

6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/v

5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes	
:*
dtype0

Adam/classifier/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä>*)
shared_nameAdam/classifier/kernel/v

,Adam/classifier/kernel/v/Read/ReadVariableOpReadVariableOpAdam/classifier/kernel/v* 
_output_shapes
:
Ä>*
dtype0

Adam/classifier/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä>*'
shared_nameAdam/classifier/bias/v
~
*Adam/classifier/bias/v/Read/ReadVariableOpReadVariableOpAdam/classifier/bias/v*
_output_shapes	
:Ä>*
dtype0

NoOpNoOp
Š¨
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ă§
valueŘ§BÔ§ BĚ§
ţ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api

$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*trainable_variables
+	variables
,	keras_api
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api

7axis
	8gamma
9beta
:moving_mean
;moving_variance
<regularization_losses
=trainable_variables
>	variables
?	keras_api
R
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api

Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
R
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
h

Wkernel
Xbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api

]axis
	^gamma
_beta
`moving_mean
amoving_variance
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
R
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
h

jkernel
kbias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api

paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
R
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
S
}regularization_losses
~trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
 	variables
Ą	keras_api
Ů
	˘iter
Łbeta_1
¤beta_2

Ľdecay
Ślearning_ratemĽmŚ%m§&m¨1mŠ2mŞ8mŤ9mŹDm­EmŽKmŻLm°WmąXm˛^mł_m´jmľkmśqmˇrm¸	mš	mş	mť	mź	m˝	mžvżvŔ%vÁ&vÂ1vĂ2vÄ8vĹ9vĆDvÇEvČKvÉLvĘWvËXvĚ^vÍ_vÎjvĎkvĐqvŃrvŇ	vÓ	vÔ	vŐ	vÖ	v×	vŘ
 
Ě
0
1
%2
&3
14
25
86
97
D8
E9
K10
L11
W12
X13
^14
_15
j16
k17
q18
r19
20
21
22
23
24
25
Ž
0
1
%2
&3
'4
(5
16
27
88
99
:10
;11
D12
E13
K14
L15
M16
N17
W18
X19
^20
_21
`22
a23
j24
k25
q26
r27
s28
t29
30
31
32
33
34
35
36
37
˛
§non_trainable_variables
 ¨layer_regularization_losses
Šlayers
Şlayer_metrics
Ťmetrics
regularization_losses
trainable_variables
	variables
 
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
˛
Źnon_trainable_variables
 ­layer_regularization_losses
Žlayers
Żlayer_metrics
°metrics
 regularization_losses
!trainable_variables
"	variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
'2
(3
˛
ąnon_trainable_variables
 ˛layer_regularization_losses
łlayers
´layer_metrics
ľmetrics
)regularization_losses
*trainable_variables
+	variables
 
 
 
˛
śnon_trainable_variables
 ˇlayer_regularization_losses
¸layers
šlayer_metrics
şmetrics
-regularization_losses
.trainable_variables
/	variables
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
˛
ťnon_trainable_variables
 źlayer_regularization_losses
˝layers
žlayer_metrics
żmetrics
3regularization_losses
4trainable_variables
5	variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
:2
;3
˛
Ŕnon_trainable_variables
 Álayer_regularization_losses
Âlayers
Ălayer_metrics
Ämetrics
<regularization_losses
=trainable_variables
>	variables
 
 
 
˛
Ĺnon_trainable_variables
 Ćlayer_regularization_losses
Çlayers
Člayer_metrics
Émetrics
@regularization_losses
Atrainable_variables
B	variables
YW
VARIABLE_VALUEconv_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

D0
E1
˛
Ęnon_trainable_variables
 Ëlayer_regularization_losses
Ělayers
Ílayer_metrics
Îmetrics
Fregularization_losses
Gtrainable_variables
H	variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1

K0
L1
M2
N3
˛
Ďnon_trainable_variables
 Đlayer_regularization_losses
Ńlayers
Ňlayer_metrics
Ómetrics
Oregularization_losses
Ptrainable_variables
Q	variables
 
 
 
˛
Ônon_trainable_variables
 Őlayer_regularization_losses
Ölayers
×layer_metrics
Řmetrics
Sregularization_losses
Ttrainable_variables
U	variables
YW
VARIABLE_VALUEconv_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

W0
X1

W0
X1
˛
Ůnon_trainable_variables
 Úlayer_regularization_losses
Űlayers
Ülayer_metrics
Ýmetrics
Yregularization_losses
Ztrainable_variables
[	variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

^0
_1
`2
a3
˛
Ţnon_trainable_variables
 ßlayer_regularization_losses
ŕlayers
álayer_metrics
âmetrics
bregularization_losses
ctrainable_variables
d	variables
 
 
 
˛
ănon_trainable_variables
 älayer_regularization_losses
ĺlayers
ćlayer_metrics
çmetrics
fregularization_losses
gtrainable_variables
h	variables
YW
VARIABLE_VALUEconv_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
˛
čnon_trainable_variables
 élayer_regularization_losses
ęlayers
ëlayer_metrics
ěmetrics
lregularization_losses
mtrainable_variables
n	variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
s2
t3
˛
ínon_trainable_variables
 îlayer_regularization_losses
ďlayers
đlayer_metrics
ńmetrics
uregularization_losses
vtrainable_variables
w	variables
 
 
 
˛
ňnon_trainable_variables
 ólayer_regularization_losses
ôlayers
őlayer_metrics
ömetrics
yregularization_losses
ztrainable_variables
{	variables
 
 
 
˛
÷non_trainable_variables
 řlayer_regularization_losses
ůlayers
úlayer_metrics
űmetrics
}regularization_losses
~trainable_variables
	variables
 
 
 
ľ
ünon_trainable_variables
 ýlayer_regularization_losses
ţlayers
˙layer_metrics
metrics
regularization_losses
trainable_variables
	variables
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
ľ
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
	variables
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
0
1
2
3
ľ
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
	variables
 
 
 
ľ
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
	variables
 
 
 
ľ
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
	variables
^\
VARIABLE_VALUEclassifier/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEclassifier/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
ľ
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
 	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
X
'0
(1
:2
;3
M4
N5
`6
a7
s8
t9
10
11
 
Ž
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
 

0
1
 
 
 
 
 

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

:0
;1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

M0
N1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

`0
a1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

s0
t1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

 total

Ącount
˘
_fn_kwargs
Ł	variables
¤	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

 0
Ą1

Ł	variables
|z
VARIABLE_VALUEAdam/conv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_4/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_4/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_5/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_5/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_5/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/classifier/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/classifier/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_4/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_4/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_5/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_5/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_5/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/classifier/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/classifier/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_imPlaceholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
dtype0*%
shape:˙˙˙˙˙˙˙˙˙ 
Ú

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_imconv_1/kernelconv_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv_2/kernelconv_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv_3/kernelconv_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv_4/kernelconv_4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv_5/kernelconv_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_1/kerneldense_1/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betaclassifier/kernelclassifier/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_45305
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ľ&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp!conv_4/kernel/Read/ReadVariableOpconv_4/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp!conv_5/kernel/Read/ReadVariableOpconv_5/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp%classifier/kernel/Read/ReadVariableOp#classifier/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv_1/kernel/m/Read/ReadVariableOp&Adam/conv_1/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp(Adam/conv_2/kernel/m/Read/ReadVariableOp&Adam/conv_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp(Adam/conv_3/kernel/m/Read/ReadVariableOp&Adam/conv_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp(Adam/conv_4/kernel/m/Read/ReadVariableOp&Adam/conv_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp(Adam/conv_5/kernel/m/Read/ReadVariableOp&Adam/conv_5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp,Adam/classifier/kernel/m/Read/ReadVariableOp*Adam/classifier/bias/m/Read/ReadVariableOp(Adam/conv_1/kernel/v/Read/ReadVariableOp&Adam/conv_1/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp(Adam/conv_2/kernel/v/Read/ReadVariableOp&Adam/conv_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp(Adam/conv_3/kernel/v/Read/ReadVariableOp&Adam/conv_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp(Adam/conv_4/kernel/v/Read/ReadVariableOp&Adam/conv_4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp(Adam/conv_5/kernel/v/Read/ReadVariableOp&Adam/conv_5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp,Adam/classifier/kernel/v/Read/ReadVariableOp*Adam/classifier/bias/v/Read/ReadVariableOpConst*p
Tini
g2e	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_47015
ä
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv_2/kernelconv_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv_3/kernelconv_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv_4/kernelconv_4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv_5/kernelconv_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_1/kerneldense_1/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceclassifier/kernelclassifier/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv_1/kernel/mAdam/conv_1/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv_2/kernel/mAdam/conv_2/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv_3/kernel/mAdam/conv_3/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv_4/kernel/mAdam/conv_4/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv_5/kernel/mAdam/conv_5/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/classifier/kernel/mAdam/classifier/bias/mAdam/conv_1/kernel/vAdam/conv_1/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv_2/kernel/vAdam/conv_2/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv_3/kernel/vAdam/conv_3/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv_4/kernel/vAdam/conv_4/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv_5/kernel/vAdam/conv_5/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/classifier/kernel/vAdam/classifier/bias/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_47322ňŤ
Ç

&__inference_conv_3_layer_call_fn_46089

inputs#
unknown:
	unknown_0:	
identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_440402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Î
Ă
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46493

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ş
Ô
5__inference_batch_normalization_3_layer_call_fn_46291

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_441132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
§Ć
Ź!
>__inference_cnn_layer_call_and_return_conditional_losses_45610

inputs?
%conv_1_conv2d_readvariableop_resource:		@4
&conv_1_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@@
%conv_2_conv2d_readvariableop_resource:@5
&conv_2_biasadd_readvariableop_resource:	<
-batch_normalization_1_readvariableop_resource:	>
/batch_normalization_1_readvariableop_1_resource:	M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	A
%conv_3_conv2d_readvariableop_resource:5
&conv_3_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	A
%conv_4_conv2d_readvariableop_resource:5
&conv_4_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	A
%conv_5_conv2d_readvariableop_resource:5
&conv_5_biasadd_readvariableop_resource:	<
-batch_normalization_4_readvariableop_resource:	>
/batch_normalization_4_readvariableop_1_resource:	M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	F
7batch_normalization_5_batchnorm_readvariableop_resource:	J
;batch_normalization_5_batchnorm_mul_readvariableop_resource:	H
9batch_normalization_5_batchnorm_readvariableop_1_resource:	H
9batch_normalization_5_batchnorm_readvariableop_2_resource:	=
)classifier_matmul_readvariableop_resource:
Ä>9
*classifier_biasadd_readvariableop_resource:	Ä>
identity˘3batch_normalization/FusedBatchNormV3/ReadVariableOp˘5batch_normalization/FusedBatchNormV3/ReadVariableOp_1˘"batch_normalization/ReadVariableOp˘$batch_normalization/ReadVariableOp_1˘5batch_normalization_1/FusedBatchNormV3/ReadVariableOp˘7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1˘$batch_normalization_1/ReadVariableOp˘&batch_normalization_1/ReadVariableOp_1˘5batch_normalization_2/FusedBatchNormV3/ReadVariableOp˘7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1˘$batch_normalization_2/ReadVariableOp˘&batch_normalization_2/ReadVariableOp_1˘5batch_normalization_3/FusedBatchNormV3/ReadVariableOp˘7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1˘$batch_normalization_3/ReadVariableOp˘&batch_normalization_3/ReadVariableOp_1˘5batch_normalization_4/FusedBatchNormV3/ReadVariableOp˘7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1˘$batch_normalization_4/ReadVariableOp˘&batch_normalization_4/ReadVariableOp_1˘.batch_normalization_5/batchnorm/ReadVariableOp˘0batch_normalization_5/batchnorm/ReadVariableOp_1˘0batch_normalization_5/batchnorm/ReadVariableOp_2˘2batch_normalization_5/batchnorm/mul/ReadVariableOp˘!classifier/BiasAdd/ReadVariableOp˘ classifier/MatMul/ReadVariableOp˘conv_1/BiasAdd/ReadVariableOp˘conv_1/Conv2D/ReadVariableOp˘conv_2/BiasAdd/ReadVariableOp˘conv_2/Conv2D/ReadVariableOp˘conv_3/BiasAdd/ReadVariableOp˘conv_3/Conv2D/ReadVariableOp˘conv_4/BiasAdd/ReadVariableOp˘conv_4/Conv2D/ReadVariableOp˘conv_5/BiasAdd/ReadVariableOp˘conv_5/Conv2D/ReadVariableOp˘dense_1/BiasAdd/ReadVariableOp˘dense_1/MatMul/ReadVariableOpŞ
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:		@*
dtype02
conv_1/Conv2D/ReadVariableOpš
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*
paddingVALID*
strides
2
conv_1/Conv2DĄ
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_1/BiasAdd/ReadVariableOp¤
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2
conv_1/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype02$
"batch_normalization/ReadVariableOpś
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype02&
$batch_normalization/ReadVariableOp_1ă
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ó
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙<@:@:@:@:@:*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2
activation/ReluŤ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_2/Conv2D/ReadVariableOpĐ
conv_2/Conv2DConv2Dactivation/Relu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
conv_2/Conv2D˘
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_2/BiasAdd/ReadVariableOpĽ
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
conv_2/BiasAddˇ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_1/ReadVariableOp˝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1ę
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpđ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ä
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
activation_1/ReluŹ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_3/Conv2D/ReadVariableOpŇ
conv_3/Conv2DConv2Dactivation_1/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
conv_3/Conv2D˘
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_3/BiasAdd/ReadVariableOpĽ
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
conv_3/BiasAddˇ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_2/ReadVariableOp˝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1ę
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpđ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ä
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
activation_2/ReluŹ
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOpŇ
conv_4/Conv2DConv2Dactivation_2/Relu:activations:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
conv_4/Conv2D˘
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_4/BiasAdd/ReadVariableOpĽ
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
conv_4/BiasAddˇ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_3/ReadVariableOp˝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1ę
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpđ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ä
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
activation_3/ReluŹ
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_5/Conv2D/ReadVariableOpŇ
conv_5/Conv2DConv2Dactivation_3/Relu:activations:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
conv_5/Conv2D˘
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_5/BiasAdd/ReadVariableOpĽ
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
conv_5/BiasAddˇ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp˝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ę
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpđ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ä
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv_5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
activation_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
activation_4/Reluž
max_pool/MaxPoolMaxPoolactivation_4/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
max_pool/MaxPool
#average_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2%
#average_pool/Mean/reduction_indicesŞ
average_pool/MeanMeanmax_pool/MaxPool:output:0,average_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
average_pool/Mean§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulaverage_pool/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMulĽ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp˘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddŐ
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_5/batchnorm/add/yá
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_5/batchnorm/addŚ
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_5/batchnorm/Rsqrtá
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpŢ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_5/batchnorm/mulË
%batch_normalization_5/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%batch_normalization_5/batchnorm/mul_1Ű
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1Ţ
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_5/batchnorm/mul_2Ű
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2Ü
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_5/batchnorm/subŢ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%batch_normalization_5/batchnorm/add_1
activation_5/ReluRelu)batch_normalization_5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
activation_5/Relu
dropout_1/IdentityIdentityactivation_5/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_1/Identity°
 classifier/MatMul/ReadVariableOpReadVariableOp)classifier_matmul_readvariableop_resource* 
_output_shapes
:
Ä>*
dtype02"
 classifier/MatMul/ReadVariableOpŞ
classifier/MatMulMatMuldropout_1/Identity:output:0(classifier/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
classifier/MatMulŽ
!classifier/BiasAdd/ReadVariableOpReadVariableOp*classifier_biasadd_readvariableop_resource*
_output_shapes	
:Ä>*
dtype02#
!classifier/BiasAdd/ReadVariableOpŽ
classifier/BiasAddBiasAddclassifier/MatMul:product:0)classifier/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
classifier/BiasAdd
classifier/SoftmaxSoftmaxclassifier/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
classifier/SoftmaxÂ
IdentityIdentityclassifier/Softmax:softmax:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp"^classifier/BiasAdd/ReadVariableOp!^classifier/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^conv_5/BiasAdd/ReadVariableOp^conv_5/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2F
!classifier/BiasAdd/ReadVariableOp!classifier/BiasAdd/ReadVariableOp2D
 classifier/MatMul/ReadVariableOp classifier/MatMul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
conv_4/BiasAdd/ReadVariableOpconv_4/BiasAdd/ReadVariableOp2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2>
conv_5/BiasAdd/ReadVariableOpconv_5/BiasAdd/ReadVariableOp2<
conv_5/Conv2D/ReadVariableOpconv_5/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ý
a
E__inference_activation_layer_call_and_return_conditional_losses_43978

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙<@:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs

c
G__inference_activation_4_layer_call_and_return_conditional_losses_44178

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs


P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43380

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3í
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43506

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3í
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ź
˝
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43172

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Î
Ă
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46187

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛p

>__inference_cnn_layer_call_and_return_conditional_losses_45216
input_im&
conv_1_45117:		@
conv_1_45119:@'
batch_normalization_45122:@'
batch_normalization_45124:@'
batch_normalization_45126:@'
batch_normalization_45128:@'
conv_2_45132:@
conv_2_45134:	*
batch_normalization_1_45137:	*
batch_normalization_1_45139:	*
batch_normalization_1_45141:	*
batch_normalization_1_45143:	(
conv_3_45147:
conv_3_45149:	*
batch_normalization_2_45152:	*
batch_normalization_2_45154:	*
batch_normalization_2_45156:	*
batch_normalization_2_45158:	(
conv_4_45162:
conv_4_45164:	*
batch_normalization_3_45167:	*
batch_normalization_3_45169:	*
batch_normalization_3_45171:	*
batch_normalization_3_45173:	(
conv_5_45177:
conv_5_45179:	*
batch_normalization_4_45182:	*
batch_normalization_4_45184:	*
batch_normalization_4_45186:	*
batch_normalization_4_45188:	!
dense_1_45194:

dense_1_45196:	*
batch_normalization_5_45199:	*
batch_normalization_5_45201:	*
batch_normalization_5_45203:	*
batch_normalization_5_45205:	$
classifier_45210:
Ä>
classifier_45212:	Ä>
identity˘+batch_normalization/StatefulPartitionedCall˘-batch_normalization_1/StatefulPartitionedCall˘-batch_normalization_2/StatefulPartitionedCall˘-batch_normalization_3/StatefulPartitionedCall˘-batch_normalization_4/StatefulPartitionedCall˘-batch_normalization_5/StatefulPartitionedCall˘"classifier/StatefulPartitionedCall˘conv_1/StatefulPartitionedCall˘conv_2/StatefulPartitionedCall˘conv_3/StatefulPartitionedCall˘conv_4/StatefulPartitionedCall˘conv_5/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘!dropout_1/StatefulPartitionedCall
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_imconv_1_45117conv_1_45119*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_439402 
conv_1/StatefulPartitionedCallŹ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_45122batch_normalization_45124batch_normalization_45126batch_normalization_45128*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_446462-
+batch_normalization/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_439782
activation/PartitionedCall°
conv_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv_2_45132conv_2_45134*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_439902 
conv_2/StatefulPartitionedCallť
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_45137batch_normalization_1_45139batch_normalization_1_45141batch_normalization_1_45143*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_445862/
-batch_normalization_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_440282
activation_1/PartitionedCall˛
conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv_3_45147conv_3_45149*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_440402 
conv_3/StatefulPartitionedCallť
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_45152batch_normalization_2_45154batch_normalization_2_45156batch_normalization_2_45158*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_445262/
-batch_normalization_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_440782
activation_2/PartitionedCall˛
conv_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv_4_45162conv_4_45164*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_440902 
conv_4/StatefulPartitionedCallť
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_45167batch_normalization_3_45169batch_normalization_3_45171batch_normalization_3_45173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_444662/
-batch_normalization_3/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_441282
activation_3/PartitionedCall˛
conv_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv_5_45177conv_5_45179*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_441402 
conv_5/StatefulPartitionedCallť
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_45182batch_normalization_4_45184batch_normalization_4_45186batch_normalization_4_45188*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_444062/
-batch_normalization_4/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_441782
activation_4/PartitionedCallţ
max_pool/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_max_pool_layer_call_and_return_conditional_losses_437422
max_pool/PartitionedCallţ
average_pool/PartitionedCallPartitionedCall!max_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_average_pool_layer_call_and_return_conditional_losses_437552
average_pool/PartitionedCallŻ
dense_1/StatefulPartitionedCallStatefulPartitionedCall%average_pool/PartitionedCall:output:0dense_1_45194dense_1_45196*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_441922!
dense_1/StatefulPartitionedCall´
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_5_45199batch_normalization_5_45201batch_normalization_5_45203batch_normalization_5_45205*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_438452/
-batch_normalization_5/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_442122
activation_5/PartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_443482#
!dropout_1/StatefulPartitionedCallĂ
"classifier/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0classifier_45210classifier_45212*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_classifier_layer_call_and_return_conditional_losses_442322$
"classifier/StatefulPartitionedCallŽ
IdentityIdentity+classifier/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall#^classifier/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2H
"classifier/StatefulPartitionedCall"classifier/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Z V
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
input_im

c
G__inference_activation_1_layer_call_and_return_conditional_losses_46080

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
ň
Ô
5__inference_batch_normalization_3_layer_call_fn_46265

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_435062
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á
	
#__inference_cnn_layer_call_fn_45386

inputs!
unknown:		@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:
Ä>

unknown_36:	Ä>
identity˘StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_cnn_layer_call_and_return_conditional_losses_442392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ź
˝
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45881

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
đ
Ô
5__inference_batch_normalization_4_layer_call_fn_46431

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436762
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°

ú
A__inference_conv_1_layer_call_and_return_conditional_losses_45793

inputs8
conv2d_readvariableop_resource:		@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Î
Ă
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43298

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ç

&__inference_conv_5_layer_call_fn_46395

inputs#
unknown:
	unknown_0:	
identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_441402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs

Ô
5__inference_batch_normalization_5_layer_call_fn_46571

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_437852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
´

ü
A__inference_conv_2_layer_call_and_return_conditional_losses_43990

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙<@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs
Ŕ

N__inference_batch_normalization_layer_call_and_return_conditional_losses_45899

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙<@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ú
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙<@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs

H
,__inference_average_pool_layer_call_fn_43761

inputs
identityŃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_average_pool_layer_call_and_return_conditional_losses_437552
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Á

ů
E__inference_classifier_layer_call_and_return_conditional_losses_44232

inputs2
matmul_readvariableop_resource:
Ä>.
biasadd_readvariableop_resource:	Ä>
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä>*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ä>*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Źp

>__inference_cnn_layer_call_and_return_conditional_losses_44852

inputs&
conv_1_44753:		@
conv_1_44755:@'
batch_normalization_44758:@'
batch_normalization_44760:@'
batch_normalization_44762:@'
batch_normalization_44764:@'
conv_2_44768:@
conv_2_44770:	*
batch_normalization_1_44773:	*
batch_normalization_1_44775:	*
batch_normalization_1_44777:	*
batch_normalization_1_44779:	(
conv_3_44783:
conv_3_44785:	*
batch_normalization_2_44788:	*
batch_normalization_2_44790:	*
batch_normalization_2_44792:	*
batch_normalization_2_44794:	(
conv_4_44798:
conv_4_44800:	*
batch_normalization_3_44803:	*
batch_normalization_3_44805:	*
batch_normalization_3_44807:	*
batch_normalization_3_44809:	(
conv_5_44813:
conv_5_44815:	*
batch_normalization_4_44818:	*
batch_normalization_4_44820:	*
batch_normalization_4_44822:	*
batch_normalization_4_44824:	!
dense_1_44830:

dense_1_44832:	*
batch_normalization_5_44835:	*
batch_normalization_5_44837:	*
batch_normalization_5_44839:	*
batch_normalization_5_44841:	$
classifier_44846:
Ä>
classifier_44848:	Ä>
identity˘+batch_normalization/StatefulPartitionedCall˘-batch_normalization_1/StatefulPartitionedCall˘-batch_normalization_2/StatefulPartitionedCall˘-batch_normalization_3/StatefulPartitionedCall˘-batch_normalization_4/StatefulPartitionedCall˘-batch_normalization_5/StatefulPartitionedCall˘"classifier/StatefulPartitionedCall˘conv_1/StatefulPartitionedCall˘conv_2/StatefulPartitionedCall˘conv_3/StatefulPartitionedCall˘conv_4/StatefulPartitionedCall˘conv_5/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘!dropout_1/StatefulPartitionedCall
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_44753conv_1_44755*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_439402 
conv_1/StatefulPartitionedCallŹ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_44758batch_normalization_44760batch_normalization_44762batch_normalization_44764*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_446462-
+batch_normalization/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_439782
activation/PartitionedCall°
conv_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv_2_44768conv_2_44770*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_439902 
conv_2/StatefulPartitionedCallť
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_44773batch_normalization_1_44775batch_normalization_1_44777batch_normalization_1_44779*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_445862/
-batch_normalization_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_440282
activation_1/PartitionedCall˛
conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv_3_44783conv_3_44785*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_440402 
conv_3/StatefulPartitionedCallť
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_44788batch_normalization_2_44790batch_normalization_2_44792batch_normalization_2_44794*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_445262/
-batch_normalization_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_440782
activation_2/PartitionedCall˛
conv_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv_4_44798conv_4_44800*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_440902 
conv_4/StatefulPartitionedCallť
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_44803batch_normalization_3_44805batch_normalization_3_44807batch_normalization_3_44809*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_444662/
-batch_normalization_3/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_441282
activation_3/PartitionedCall˛
conv_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv_5_44813conv_5_44815*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_441402 
conv_5/StatefulPartitionedCallť
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_44818batch_normalization_4_44820batch_normalization_4_44822batch_normalization_4_44824*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_444062/
-batch_normalization_4/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_441782
activation_4/PartitionedCallţ
max_pool/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_max_pool_layer_call_and_return_conditional_losses_437422
max_pool/PartitionedCallţ
average_pool/PartitionedCallPartitionedCall!max_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_average_pool_layer_call_and_return_conditional_losses_437552
average_pool/PartitionedCallŻ
dense_1/StatefulPartitionedCallStatefulPartitionedCall%average_pool/PartitionedCall:output:0dense_1_44830dense_1_44832*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_441922!
dense_1/StatefulPartitionedCall´
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_5_44835batch_normalization_5_44837batch_normalization_5_44839batch_normalization_5_44841*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_438452/
-batch_normalization_5/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_442122
activation_5/PartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_443482#
!dropout_1/StatefulPartitionedCallĂ
"classifier/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0classifier_44846classifier_44848*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_classifier_layer_call_and_return_conditional_losses_442322$
"classifier/StatefulPartitionedCallŽ
IdentityIdentity+classifier/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall#^classifier/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2H
"classifier/StatefulPartitionedCall"classifier/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ů	
ö
B__inference_dense_1_layer_call_and_return_conditional_losses_46558

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ä
Î
3__inference_batch_normalization_layer_call_fn_45819

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity˘StatefulPartitionedCallł
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_431722
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs


P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43632

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3í
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸

ý
A__inference_conv_3_layer_call_and_return_conditional_losses_44040

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ů	
ö
B__inference_dense_1_layer_call_and_return_conditional_losses_44192

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ô
˝
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44646

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ř
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙<@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ţ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙<@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs
×
c
G__inference_average_pool_layer_call_and_return_conditional_losses_43755

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç
	
#__inference_cnn_layer_call_fn_44318
input_im!
unknown:		@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:
Ä>

unknown_36:	Ä>
identity˘StatefulPartitionedCallă
StatefulPartitionedCallStatefulPartitionedCallinput_imunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_cnn_layer_call_and_return_conditional_losses_442392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
input_im
á
F
*__inference_activation_layer_call_fn_45922

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_439782
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙<@:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs
Ş
Ô
5__inference_batch_normalization_2_layer_call_fn_46138

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_440632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ç

&__inference_conv_4_layer_call_fn_46242

inputs#
unknown:
	unknown_0:	
identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_440902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
¸

ý
A__inference_conv_3_layer_call_and_return_conditional_losses_46099

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ň

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44163

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ű
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
đ
Ô
5__inference_batch_normalization_3_layer_call_fn_46278

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_435502
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸

ý
A__inference_conv_4_layer_call_and_return_conditional_losses_46252

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
É
%
>__inference_cnn_layer_call_and_return_conditional_losses_45774

inputs?
%conv_1_conv2d_readvariableop_resource:		@4
&conv_1_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@@
%conv_2_conv2d_readvariableop_resource:@5
&conv_2_biasadd_readvariableop_resource:	<
-batch_normalization_1_readvariableop_resource:	>
/batch_normalization_1_readvariableop_1_resource:	M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	A
%conv_3_conv2d_readvariableop_resource:5
&conv_3_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	A
%conv_4_conv2d_readvariableop_resource:5
&conv_4_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	A
%conv_5_conv2d_readvariableop_resource:5
&conv_5_biasadd_readvariableop_resource:	<
-batch_normalization_4_readvariableop_resource:	>
/batch_normalization_4_readvariableop_1_resource:	M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	L
=batch_normalization_5_assignmovingavg_readvariableop_resource:	N
?batch_normalization_5_assignmovingavg_1_readvariableop_resource:	J
;batch_normalization_5_batchnorm_mul_readvariableop_resource:	F
7batch_normalization_5_batchnorm_readvariableop_resource:	=
)classifier_matmul_readvariableop_resource:
Ä>9
*classifier_biasadd_readvariableop_resource:	Ä>
identity˘"batch_normalization/AssignNewValue˘$batch_normalization/AssignNewValue_1˘3batch_normalization/FusedBatchNormV3/ReadVariableOp˘5batch_normalization/FusedBatchNormV3/ReadVariableOp_1˘"batch_normalization/ReadVariableOp˘$batch_normalization/ReadVariableOp_1˘$batch_normalization_1/AssignNewValue˘&batch_normalization_1/AssignNewValue_1˘5batch_normalization_1/FusedBatchNormV3/ReadVariableOp˘7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1˘$batch_normalization_1/ReadVariableOp˘&batch_normalization_1/ReadVariableOp_1˘$batch_normalization_2/AssignNewValue˘&batch_normalization_2/AssignNewValue_1˘5batch_normalization_2/FusedBatchNormV3/ReadVariableOp˘7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1˘$batch_normalization_2/ReadVariableOp˘&batch_normalization_2/ReadVariableOp_1˘$batch_normalization_3/AssignNewValue˘&batch_normalization_3/AssignNewValue_1˘5batch_normalization_3/FusedBatchNormV3/ReadVariableOp˘7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1˘$batch_normalization_3/ReadVariableOp˘&batch_normalization_3/ReadVariableOp_1˘$batch_normalization_4/AssignNewValue˘&batch_normalization_4/AssignNewValue_1˘5batch_normalization_4/FusedBatchNormV3/ReadVariableOp˘7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1˘$batch_normalization_4/ReadVariableOp˘&batch_normalization_4/ReadVariableOp_1˘%batch_normalization_5/AssignMovingAvg˘4batch_normalization_5/AssignMovingAvg/ReadVariableOp˘'batch_normalization_5/AssignMovingAvg_1˘6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp˘.batch_normalization_5/batchnorm/ReadVariableOp˘2batch_normalization_5/batchnorm/mul/ReadVariableOp˘!classifier/BiasAdd/ReadVariableOp˘ classifier/MatMul/ReadVariableOp˘conv_1/BiasAdd/ReadVariableOp˘conv_1/Conv2D/ReadVariableOp˘conv_2/BiasAdd/ReadVariableOp˘conv_2/Conv2D/ReadVariableOp˘conv_3/BiasAdd/ReadVariableOp˘conv_3/Conv2D/ReadVariableOp˘conv_4/BiasAdd/ReadVariableOp˘conv_4/Conv2D/ReadVariableOp˘conv_5/BiasAdd/ReadVariableOp˘conv_5/Conv2D/ReadVariableOp˘dense_1/BiasAdd/ReadVariableOp˘dense_1/MatMul/ReadVariableOpŞ
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:		@*
dtype02
conv_1/Conv2D/ReadVariableOpš
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*
paddingVALID*
strides
2
conv_1/Conv2DĄ
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_1/BiasAdd/ReadVariableOp¤
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2
conv_1/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype02$
"batch_normalization/ReadVariableOpś
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype02&
$batch_normalization/ReadVariableOp_1ă
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1á
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙<@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2&
$batch_normalization/FusedBatchNormV3Ś
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue˛
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2
activation/ReluŤ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_2/Conv2D/ReadVariableOpĐ
conv_2/Conv2DConv2Dactivation/Relu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
conv_2/Conv2D˘
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_2/BiasAdd/ReadVariableOpĽ
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
conv_2/BiasAddˇ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_1/ReadVariableOp˝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1ę
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpđ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ň
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_1/FusedBatchNormV3°
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueź
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
activation_1/ReluŹ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_3/Conv2D/ReadVariableOpŇ
conv_3/Conv2DConv2Dactivation_1/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
conv_3/Conv2D˘
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_3/BiasAdd/ReadVariableOpĽ
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
conv_3/BiasAddˇ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_2/ReadVariableOp˝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1ę
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpđ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ň
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_2/FusedBatchNormV3°
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueź
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
activation_2/ReluŹ
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOpŇ
conv_4/Conv2DConv2Dactivation_2/Relu:activations:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
conv_4/Conv2D˘
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_4/BiasAdd/ReadVariableOpĽ
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
conv_4/BiasAddˇ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_3/ReadVariableOp˝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1ę
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpđ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ň
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_3/FusedBatchNormV3°
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValueź
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
activation_3/ReluŹ
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_5/Conv2D/ReadVariableOpŇ
conv_5/Conv2DConv2Dactivation_3/Relu:activations:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
conv_5/Conv2D˘
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_5/BiasAdd/ReadVariableOpĽ
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
conv_5/BiasAddˇ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp˝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ę
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpđ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ň
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv_5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_4/FusedBatchNormV3°
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValueź
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1
activation_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
activation_4/Reluž
max_pool/MaxPoolMaxPoolactivation_4/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
max_pool/MaxPool
#average_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2%
#average_pool/Mean/reduction_indicesŞ
average_pool/MeanMeanmax_pool/MaxPool:output:0,average_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
average_pool/Mean§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulaverage_pool/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMulĽ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp˘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddś
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indicesä
"batch_normalization_5/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2$
"batch_normalization_5/moments/meanż
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:	2,
*batch_normalization_5/moments/StopGradientů
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_5/moments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙21
/batch_normalization_5/moments/SquaredDifferencež
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_5/moments/variance/reduction_indices
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2(
&batch_normalization_5/moments/varianceĂ
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2'
%batch_normalization_5/moments/SqueezeË
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1
+batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_5/AssignMovingAvg/decayç
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOpń
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes	
:2+
)batch_normalization_5/AssignMovingAvg/subč
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2+
)batch_normalization_5/AssignMovingAvg/mul­
%batch_normalization_5/AssignMovingAvgAssignSubVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_5/AssignMovingAvgŁ
-batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_5/AssignMovingAvg_1/decayí
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpů
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2-
+batch_normalization_5/AssignMovingAvg_1/subđ
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2-
+batch_normalization_5/AssignMovingAvg_1/mulˇ
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_5/AssignMovingAvg_1
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_5/batchnorm/add/yŰ
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_5/batchnorm/addŚ
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_5/batchnorm/Rsqrtá
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpŢ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_5/batchnorm/mulË
%batch_normalization_5/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%batch_normalization_5/batchnorm/mul_1Ô
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_5/batchnorm/mul_2Ő
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpÚ
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_5/batchnorm/subŢ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%batch_normalization_5/batchnorm/add_1
activation_5/ReluRelu)batch_normalization_5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
activation_5/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/dropout/ConstŤ
dropout_1/dropout/MulMulactivation_5/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShapeactivation_5/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeß
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*

seed20
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>2"
 dropout_1/dropout/GreaterEqual/yç
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_1/dropout/CastŁ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_1/dropout/Mul_1°
 classifier/MatMul/ReadVariableOpReadVariableOp)classifier_matmul_readvariableop_resource* 
_output_shapes
:
Ä>*
dtype02"
 classifier/MatMul/ReadVariableOpŞ
classifier/MatMulMatMuldropout_1/dropout/Mul_1:z:0(classifier/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
classifier/MatMulŽ
!classifier/BiasAdd/ReadVariableOpReadVariableOp*classifier_biasadd_readvariableop_resource*
_output_shapes	
:Ä>*
dtype02#
!classifier/BiasAdd/ReadVariableOpŽ
classifier/BiasAddBiasAddclassifier/MatMul:product:0)classifier/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
classifier/BiasAdd
classifier/SoftmaxSoftmaxclassifier/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
classifier/SoftmaxŞ
IdentityIdentityclassifier/Softmax:softmax:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp"^classifier/BiasAdd/ReadVariableOp!^classifier/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^conv_5/BiasAdd/ReadVariableOp^conv_5/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12N
%batch_normalization_5/AssignMovingAvg%batch_normalization_5/AssignMovingAvg2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_1'batch_normalization_5/AssignMovingAvg_12p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2F
!classifier/BiasAdd/ReadVariableOp!classifier/BiasAdd/ReadVariableOp2D
 classifier/MatMul/ReadVariableOp classifier/MatMul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
conv_4/BiasAdd/ReadVariableOpconv_4/BiasAdd/ReadVariableOp2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2>
conv_5/BiasAdd/ReadVariableOpconv_5/BiasAdd/ReadVariableOp2<
conv_5/Conv2D/ReadVariableOpconv_5/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Î
Ă
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43550

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
Ă
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43676

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
é
H
,__inference_activation_2_layer_call_fn_46228

inputs
identityŃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_440782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46322

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3í
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


N__inference_batch_normalization_layer_call_and_return_conditional_losses_45863

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ě
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
é
H
,__inference_activation_3_layer_call_fn_46381

inputs
identityŃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_441282
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
¨
Ô
5__inference_batch_normalization_4_layer_call_fn_46457

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_444062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
ęÂ
Ń,
__inference__traced_save_47015
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop,
(savev2_conv_4_kernel_read_readvariableop*
&savev2_conv_4_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop,
(savev2_conv_5_kernel_read_readvariableop*
&savev2_conv_5_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop0
,savev2_classifier_kernel_read_readvariableop.
*savev2_classifier_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv_1_kernel_m_read_readvariableop1
-savev2_adam_conv_1_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop3
/savev2_adam_conv_2_kernel_m_read_readvariableop1
-savev2_adam_conv_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop3
/savev2_adam_conv_3_kernel_m_read_readvariableop1
-savev2_adam_conv_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop3
/savev2_adam_conv_4_kernel_m_read_readvariableop1
-savev2_adam_conv_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop3
/savev2_adam_conv_5_kernel_m_read_readvariableop1
-savev2_adam_conv_5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop7
3savev2_adam_classifier_kernel_m_read_readvariableop5
1savev2_adam_classifier_bias_m_read_readvariableop3
/savev2_adam_conv_1_kernel_v_read_readvariableop1
-savev2_adam_conv_1_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop3
/savev2_adam_conv_2_kernel_v_read_readvariableop1
-savev2_adam_conv_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop3
/savev2_adam_conv_3_kernel_v_read_readvariableop1
-savev2_adam_conv_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop3
/savev2_adam_conv_4_kernel_v_read_readvariableop1
-savev2_adam_conv_4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop3
/savev2_adam_conv_5_kernel_v_read_readvariableop1
-savev2_adam_conv_5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop7
3savev2_adam_classifier_kernel_v_read_readvariableop5
1savev2_adam_classifier_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĆ7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ř6
valueÎ6BË6dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÓ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ý
valueÓBĐdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices÷*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop(savev2_conv_5_kernel_read_readvariableop&savev2_conv_5_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop,savev2_classifier_kernel_read_readvariableop*savev2_classifier_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv_1_kernel_m_read_readvariableop-savev2_adam_conv_1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop/savev2_adam_conv_2_kernel_m_read_readvariableop-savev2_adam_conv_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop/savev2_adam_conv_3_kernel_m_read_readvariableop-savev2_adam_conv_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop/savev2_adam_conv_4_kernel_m_read_readvariableop-savev2_adam_conv_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop/savev2_adam_conv_5_kernel_m_read_readvariableop-savev2_adam_conv_5_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop3savev2_adam_classifier_kernel_m_read_readvariableop1savev2_adam_classifier_bias_m_read_readvariableop/savev2_adam_conv_1_kernel_v_read_readvariableop-savev2_adam_conv_1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop/savev2_adam_conv_2_kernel_v_read_readvariableop-savev2_adam_conv_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop/savev2_adam_conv_3_kernel_v_read_readvariableop-savev2_adam_conv_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop/savev2_adam_conv_4_kernel_v_read_readvariableop-savev2_adam_conv_4_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop/savev2_adam_conv_5_kernel_v_read_readvariableop-savev2_adam_conv_5_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop3savev2_adam_classifier_kernel_v_read_readvariableop1savev2_adam_classifier_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*î
_input_shapesÜ
Ů: :		@:@:@:@:@:@:@::::::::::::::::::::::::
::::::
Ä>:Ä>: : : : : : : : : :		@:@:@:@:@::::::::::::::::
::::
Ä>:Ä>:		@:@:@:@:@::::::::::::::::
::::
Ä>:Ä>: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::!$

_output_shapes	
::&%"
 
_output_shapes
:
Ä>:!&

_output_shapes	
:Ä>:'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :,0(
&
_output_shapes
:		@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:-4)
'
_output_shapes
:@:!5

_output_shapes	
::!6

_output_shapes	
::!7

_output_shapes	
::.8*
(
_output_shapes
::!9

_output_shapes	
::!:

_output_shapes	
::!;

_output_shapes	
::.<*
(
_output_shapes
::!=

_output_shapes	
::!>

_output_shapes	
::!?

_output_shapes	
::.@*
(
_output_shapes
::!A

_output_shapes	
::!B

_output_shapes	
::!C

_output_shapes	
::&D"
 
_output_shapes
:
:!E

_output_shapes	
::!F

_output_shapes	
::!G

_output_shapes	
::&H"
 
_output_shapes
:
Ä>:!I

_output_shapes	
:Ä>:,J(
&
_output_shapes
:		@: K

_output_shapes
:@: L

_output_shapes
:@: M

_output_shapes
:@:-N)
'
_output_shapes
:@:!O

_output_shapes	
::!P

_output_shapes	
::!Q

_output_shapes	
::.R*
(
_output_shapes
::!S

_output_shapes	
::!T

_output_shapes	
::!U

_output_shapes	
::.V*
(
_output_shapes
::!W

_output_shapes	
::!X

_output_shapes	
::!Y

_output_shapes	
::.Z*
(
_output_shapes
::![

_output_shapes	
::!\

_output_shapes	
::!]

_output_shapes	
::&^"
 
_output_shapes
:
:!_

_output_shapes	
::!`

_output_shapes	
::!a

_output_shapes	
::&b"
 
_output_shapes
:
Ä>:!c

_output_shapes	
:Ä>:d

_output_shapes
: 
ň
Ô
5__inference_batch_normalization_4_layer_call_fn_46418

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436322
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸

ý
A__inference_conv_5_layer_call_and_return_conditional_losses_44140

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
¨
Ô
5__inference_batch_normalization_3_layer_call_fn_46304

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_444662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Á

ů
E__inference_classifier_layer_call_and_return_conditional_losses_46695

inputs2
matmul_readvariableop_resource:
Ä>.
biasadd_readvariableop_resource:	Ä>
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä>*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ä>*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
Ă
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46340

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ă
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46376

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1˙
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
É
	
#__inference_signature_wrapper_45305
input_im!
unknown:		@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:
Ä>

unknown_36:	Ä>
identity˘StatefulPartitionedCallĹ
StatefulPartitionedCallStatefulPartitionedCallinput_imunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_431062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
input_im
ô
˝
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45917

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ř
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙<@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ţ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙<@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs

Î
3__inference_batch_normalization_layer_call_fn_45845

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity˘StatefulPartitionedCallĄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_446462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙<@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs
ň
Ô
5__inference_batch_normalization_2_layer_call_fn_46112

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_433802
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ă
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44466

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1˙
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ű
	
#__inference_cnn_layer_call_fn_45012
input_im!
unknown:		@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:
Ä>

unknown_36:	Ä>
identity˘StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinput_imunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*<
_read_only_resource_inputs
	
 #$%&*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_cnn_layer_call_and_return_conditional_losses_448522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
input_im
ă
c
G__inference_activation_5_layer_call_and_return_conditional_losses_44212

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨
Ô
5__inference_batch_normalization_1_layer_call_fn_45998

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_445862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs

c
G__inference_activation_4_layer_call_and_return_conditional_losses_46539

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ę
D
(__inference_max_pool_layer_call_fn_43748

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_max_pool_layer_call_and_return_conditional_losses_437422
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
äŹ
ĹA
!__inference__traced_restore_47322
file_prefix8
assignvariableop_conv_1_kernel:		@,
assignvariableop_1_conv_1_bias:@:
,assignvariableop_2_batch_normalization_gamma:@9
+assignvariableop_3_batch_normalization_beta:@@
2assignvariableop_4_batch_normalization_moving_mean:@D
6assignvariableop_5_batch_normalization_moving_variance:@;
 assignvariableop_6_conv_2_kernel:@-
assignvariableop_7_conv_2_bias:	=
.assignvariableop_8_batch_normalization_1_gamma:	<
-assignvariableop_9_batch_normalization_1_beta:	D
5assignvariableop_10_batch_normalization_1_moving_mean:	H
9assignvariableop_11_batch_normalization_1_moving_variance:	=
!assignvariableop_12_conv_3_kernel:.
assignvariableop_13_conv_3_bias:	>
/assignvariableop_14_batch_normalization_2_gamma:	=
.assignvariableop_15_batch_normalization_2_beta:	D
5assignvariableop_16_batch_normalization_2_moving_mean:	H
9assignvariableop_17_batch_normalization_2_moving_variance:	=
!assignvariableop_18_conv_4_kernel:.
assignvariableop_19_conv_4_bias:	>
/assignvariableop_20_batch_normalization_3_gamma:	=
.assignvariableop_21_batch_normalization_3_beta:	D
5assignvariableop_22_batch_normalization_3_moving_mean:	H
9assignvariableop_23_batch_normalization_3_moving_variance:	=
!assignvariableop_24_conv_5_kernel:.
assignvariableop_25_conv_5_bias:	>
/assignvariableop_26_batch_normalization_4_gamma:	=
.assignvariableop_27_batch_normalization_4_beta:	D
5assignvariableop_28_batch_normalization_4_moving_mean:	H
9assignvariableop_29_batch_normalization_4_moving_variance:	6
"assignvariableop_30_dense_1_kernel:
/
 assignvariableop_31_dense_1_bias:	>
/assignvariableop_32_batch_normalization_5_gamma:	=
.assignvariableop_33_batch_normalization_5_beta:	D
5assignvariableop_34_batch_normalization_5_moving_mean:	H
9assignvariableop_35_batch_normalization_5_moving_variance:	9
%assignvariableop_36_classifier_kernel:
Ä>2
#assignvariableop_37_classifier_bias:	Ä>'
assignvariableop_38_adam_iter:	 )
assignvariableop_39_adam_beta_1: )
assignvariableop_40_adam_beta_2: (
assignvariableop_41_adam_decay: 0
&assignvariableop_42_adam_learning_rate: #
assignvariableop_43_total: #
assignvariableop_44_count: %
assignvariableop_45_total_1: %
assignvariableop_46_count_1: B
(assignvariableop_47_adam_conv_1_kernel_m:		@4
&assignvariableop_48_adam_conv_1_bias_m:@B
4assignvariableop_49_adam_batch_normalization_gamma_m:@A
3assignvariableop_50_adam_batch_normalization_beta_m:@C
(assignvariableop_51_adam_conv_2_kernel_m:@5
&assignvariableop_52_adam_conv_2_bias_m:	E
6assignvariableop_53_adam_batch_normalization_1_gamma_m:	D
5assignvariableop_54_adam_batch_normalization_1_beta_m:	D
(assignvariableop_55_adam_conv_3_kernel_m:5
&assignvariableop_56_adam_conv_3_bias_m:	E
6assignvariableop_57_adam_batch_normalization_2_gamma_m:	D
5assignvariableop_58_adam_batch_normalization_2_beta_m:	D
(assignvariableop_59_adam_conv_4_kernel_m:5
&assignvariableop_60_adam_conv_4_bias_m:	E
6assignvariableop_61_adam_batch_normalization_3_gamma_m:	D
5assignvariableop_62_adam_batch_normalization_3_beta_m:	D
(assignvariableop_63_adam_conv_5_kernel_m:5
&assignvariableop_64_adam_conv_5_bias_m:	E
6assignvariableop_65_adam_batch_normalization_4_gamma_m:	D
5assignvariableop_66_adam_batch_normalization_4_beta_m:	=
)assignvariableop_67_adam_dense_1_kernel_m:
6
'assignvariableop_68_adam_dense_1_bias_m:	E
6assignvariableop_69_adam_batch_normalization_5_gamma_m:	D
5assignvariableop_70_adam_batch_normalization_5_beta_m:	@
,assignvariableop_71_adam_classifier_kernel_m:
Ä>9
*assignvariableop_72_adam_classifier_bias_m:	Ä>B
(assignvariableop_73_adam_conv_1_kernel_v:		@4
&assignvariableop_74_adam_conv_1_bias_v:@B
4assignvariableop_75_adam_batch_normalization_gamma_v:@A
3assignvariableop_76_adam_batch_normalization_beta_v:@C
(assignvariableop_77_adam_conv_2_kernel_v:@5
&assignvariableop_78_adam_conv_2_bias_v:	E
6assignvariableop_79_adam_batch_normalization_1_gamma_v:	D
5assignvariableop_80_adam_batch_normalization_1_beta_v:	D
(assignvariableop_81_adam_conv_3_kernel_v:5
&assignvariableop_82_adam_conv_3_bias_v:	E
6assignvariableop_83_adam_batch_normalization_2_gamma_v:	D
5assignvariableop_84_adam_batch_normalization_2_beta_v:	D
(assignvariableop_85_adam_conv_4_kernel_v:5
&assignvariableop_86_adam_conv_4_bias_v:	E
6assignvariableop_87_adam_batch_normalization_3_gamma_v:	D
5assignvariableop_88_adam_batch_normalization_3_beta_v:	D
(assignvariableop_89_adam_conv_5_kernel_v:5
&assignvariableop_90_adam_conv_5_bias_v:	E
6assignvariableop_91_adam_batch_normalization_4_gamma_v:	D
5assignvariableop_92_adam_batch_normalization_4_beta_v:	=
)assignvariableop_93_adam_dense_1_kernel_v:
6
'assignvariableop_94_adam_dense_1_bias_v:	E
6assignvariableop_95_adam_batch_normalization_5_gamma_v:	D
5assignvariableop_96_adam_batch_normalization_5_beta_v:	@
,assignvariableop_97_adam_classifier_kernel_v:
Ä>9
*assignvariableop_98_adam_classifier_bias_v:	Ä>
identity_100˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_33˘AssignVariableOp_34˘AssignVariableOp_35˘AssignVariableOp_36˘AssignVariableOp_37˘AssignVariableOp_38˘AssignVariableOp_39˘AssignVariableOp_4˘AssignVariableOp_40˘AssignVariableOp_41˘AssignVariableOp_42˘AssignVariableOp_43˘AssignVariableOp_44˘AssignVariableOp_45˘AssignVariableOp_46˘AssignVariableOp_47˘AssignVariableOp_48˘AssignVariableOp_49˘AssignVariableOp_5˘AssignVariableOp_50˘AssignVariableOp_51˘AssignVariableOp_52˘AssignVariableOp_53˘AssignVariableOp_54˘AssignVariableOp_55˘AssignVariableOp_56˘AssignVariableOp_57˘AssignVariableOp_58˘AssignVariableOp_59˘AssignVariableOp_6˘AssignVariableOp_60˘AssignVariableOp_61˘AssignVariableOp_62˘AssignVariableOp_63˘AssignVariableOp_64˘AssignVariableOp_65˘AssignVariableOp_66˘AssignVariableOp_67˘AssignVariableOp_68˘AssignVariableOp_69˘AssignVariableOp_7˘AssignVariableOp_70˘AssignVariableOp_71˘AssignVariableOp_72˘AssignVariableOp_73˘AssignVariableOp_74˘AssignVariableOp_75˘AssignVariableOp_76˘AssignVariableOp_77˘AssignVariableOp_78˘AssignVariableOp_79˘AssignVariableOp_8˘AssignVariableOp_80˘AssignVariableOp_81˘AssignVariableOp_82˘AssignVariableOp_83˘AssignVariableOp_84˘AssignVariableOp_85˘AssignVariableOp_86˘AssignVariableOp_87˘AssignVariableOp_88˘AssignVariableOp_89˘AssignVariableOp_9˘AssignVariableOp_90˘AssignVariableOp_91˘AssignVariableOp_92˘AssignVariableOp_93˘AssignVariableOp_94˘AssignVariableOp_95˘AssignVariableOp_96˘AssignVariableOp_97˘AssignVariableOp_98Ě7
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ř6
valueÎ6BË6dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesŮ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ý
valueÓBĐdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices˘
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ł
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ą
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3°
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ˇ
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ť
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ľ
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ł
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ł
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9˛
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10˝
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Š
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ˇ
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ś
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16˝
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Á
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Š
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19§
AssignVariableOp_19AssignVariableOpassignvariableop_19_conv_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ˇ
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ś
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22˝
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Á
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Š
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25§
AssignVariableOp_25AssignVariableOpassignvariableop_25_conv_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ˇ
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ś
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28˝
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Á
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ş
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¨
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ˇ
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ś
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34˝
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Á
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36­
AssignVariableOp_36AssignVariableOp%assignvariableop_36_classifier_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ť
AssignVariableOp_37AssignVariableOp#assignvariableop_37_classifier_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_38Ľ
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39§
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40§
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ś
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ž
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ą
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ą
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ł
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ł
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47°
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ž
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_conv_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49ź
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_batch_normalization_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50ť
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_batch_normalization_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51°
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_conv_2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ž
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_conv_2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53ž
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_1_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54˝
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_1_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55°
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv_3_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Ž
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_conv_3_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ž
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_2_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58˝
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_2_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59°
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_conv_4_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ž
AssignVariableOp_60AssignVariableOp&assignvariableop_60_adam_conv_4_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61ž
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_3_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62˝
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_3_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63°
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_conv_5_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Ž
AssignVariableOp_64AssignVariableOp&assignvariableop_64_adam_conv_5_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65ž
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_4_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66˝
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_4_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67ą
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_dense_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Ż
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_dense_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69ž
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_5_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70˝
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_5_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71´
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_classifier_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72˛
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_classifier_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73°
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_conv_1_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Ž
AssignVariableOp_74AssignVariableOp&assignvariableop_74_adam_conv_1_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75ź
AssignVariableOp_75AssignVariableOp4assignvariableop_75_adam_batch_normalization_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76ť
AssignVariableOp_76AssignVariableOp3assignvariableop_76_adam_batch_normalization_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77°
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_conv_2_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Ž
AssignVariableOp_78AssignVariableOp&assignvariableop_78_adam_conv_2_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79ž
AssignVariableOp_79AssignVariableOp6assignvariableop_79_adam_batch_normalization_1_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80˝
AssignVariableOp_80AssignVariableOp5assignvariableop_80_adam_batch_normalization_1_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81°
AssignVariableOp_81AssignVariableOp(assignvariableop_81_adam_conv_3_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82Ž
AssignVariableOp_82AssignVariableOp&assignvariableop_82_adam_conv_3_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83ž
AssignVariableOp_83AssignVariableOp6assignvariableop_83_adam_batch_normalization_2_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84˝
AssignVariableOp_84AssignVariableOp5assignvariableop_84_adam_batch_normalization_2_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85°
AssignVariableOp_85AssignVariableOp(assignvariableop_85_adam_conv_4_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86Ž
AssignVariableOp_86AssignVariableOp&assignvariableop_86_adam_conv_4_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87ž
AssignVariableOp_87AssignVariableOp6assignvariableop_87_adam_batch_normalization_3_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88˝
AssignVariableOp_88AssignVariableOp5assignvariableop_88_adam_batch_normalization_3_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89°
AssignVariableOp_89AssignVariableOp(assignvariableop_89_adam_conv_5_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Ž
AssignVariableOp_90AssignVariableOp&assignvariableop_90_adam_conv_5_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91ž
AssignVariableOp_91AssignVariableOp6assignvariableop_91_adam_batch_normalization_4_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92˝
AssignVariableOp_92AssignVariableOp5assignvariableop_92_adam_batch_normalization_4_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93ą
AssignVariableOp_93AssignVariableOp)assignvariableop_93_adam_dense_1_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94Ż
AssignVariableOp_94AssignVariableOp'assignvariableop_94_adam_dense_1_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95ž
AssignVariableOp_95AssignVariableOp6assignvariableop_95_adam_batch_normalization_5_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96˝
AssignVariableOp_96AssignVariableOp5assignvariableop_96_adam_batch_normalization_5_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97´
AssignVariableOp_97AssignVariableOp,assignvariableop_97_adam_classifier_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98˛
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adam_classifier_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_989
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpŕ
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_99Ő
Identity_100IdentityIdentity_99:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*
T0*
_output_shapes
: 2
Identity_100"%
identity_100Identity_100:output:0*Ý
_input_shapesË
Č: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Î
Ă
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46034

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ý
a
E__inference_activation_layer_call_and_return_conditional_losses_45927

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙<@:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs
¸

ý
A__inference_conv_4_layer_call_and_return_conditional_losses_44090

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Î
Ă
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43424

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°

ú
A__inference_conv_1_layer_call_and_return_conditional_losses_43940

inputs8
conv2d_readvariableop_resource:		@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ň
Ô
5__inference_batch_normalization_1_layer_call_fn_45959

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_432542
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46511

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ű
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs


P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46169

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3í
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
´

ü
A__inference_conv_2_layer_call_and_return_conditional_losses_45946

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙<@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43254

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3í
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

c
G__inference_activation_3_layer_call_and_return_conditional_losses_46386

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
¸

ý
A__inference_conv_5_layer_call_and_return_conditional_losses_46405

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs

c
G__inference_activation_2_layer_call_and_return_conditional_losses_46233

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ż
ł
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43785

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity˘batchnorm/ReadVariableOp˘batchnorm/ReadVariableOp_1˘batchnorm/ReadVariableOp_2˘batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Â

&__inference_conv_1_layer_call_fn_45783

inputs!
unknown:		@
	unknown_0:@
identity˘StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_439402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙ : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ż
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_46675

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

c
G__inference_activation_3_layer_call_and_return_conditional_losses_44128

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
o
ä
>__inference_cnn_layer_call_and_return_conditional_losses_45114
input_im&
conv_1_45015:		@
conv_1_45017:@'
batch_normalization_45020:@'
batch_normalization_45022:@'
batch_normalization_45024:@'
batch_normalization_45026:@'
conv_2_45030:@
conv_2_45032:	*
batch_normalization_1_45035:	*
batch_normalization_1_45037:	*
batch_normalization_1_45039:	*
batch_normalization_1_45041:	(
conv_3_45045:
conv_3_45047:	*
batch_normalization_2_45050:	*
batch_normalization_2_45052:	*
batch_normalization_2_45054:	*
batch_normalization_2_45056:	(
conv_4_45060:
conv_4_45062:	*
batch_normalization_3_45065:	*
batch_normalization_3_45067:	*
batch_normalization_3_45069:	*
batch_normalization_3_45071:	(
conv_5_45075:
conv_5_45077:	*
batch_normalization_4_45080:	*
batch_normalization_4_45082:	*
batch_normalization_4_45084:	*
batch_normalization_4_45086:	!
dense_1_45092:

dense_1_45094:	*
batch_normalization_5_45097:	*
batch_normalization_5_45099:	*
batch_normalization_5_45101:	*
batch_normalization_5_45103:	$
classifier_45108:
Ä>
classifier_45110:	Ä>
identity˘+batch_normalization/StatefulPartitionedCall˘-batch_normalization_1/StatefulPartitionedCall˘-batch_normalization_2/StatefulPartitionedCall˘-batch_normalization_3/StatefulPartitionedCall˘-batch_normalization_4/StatefulPartitionedCall˘-batch_normalization_5/StatefulPartitionedCall˘"classifier/StatefulPartitionedCall˘conv_1/StatefulPartitionedCall˘conv_2/StatefulPartitionedCall˘conv_3/StatefulPartitionedCall˘conv_4/StatefulPartitionedCall˘conv_5/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_imconv_1_45015conv_1_45017*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_439402 
conv_1/StatefulPartitionedCallŽ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_45020batch_normalization_45022batch_normalization_45024batch_normalization_45026*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_439632-
+batch_normalization/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_439782
activation/PartitionedCall°
conv_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv_2_45030conv_2_45032*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_439902 
conv_2/StatefulPartitionedCall˝
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_45035batch_normalization_1_45037batch_normalization_1_45039batch_normalization_1_45041*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_440132/
-batch_normalization_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_440282
activation_1/PartitionedCall˛
conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv_3_45045conv_3_45047*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_440402 
conv_3/StatefulPartitionedCall˝
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_45050batch_normalization_2_45052batch_normalization_2_45054batch_normalization_2_45056*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_440632/
-batch_normalization_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_440782
activation_2/PartitionedCall˛
conv_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv_4_45060conv_4_45062*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_440902 
conv_4/StatefulPartitionedCall˝
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_45065batch_normalization_3_45067batch_normalization_3_45069batch_normalization_3_45071*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_441132/
-batch_normalization_3/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_441282
activation_3/PartitionedCall˛
conv_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv_5_45075conv_5_45077*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_441402 
conv_5/StatefulPartitionedCall˝
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_45080batch_normalization_4_45082batch_normalization_4_45084batch_normalization_4_45086*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_441632/
-batch_normalization_4/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_441782
activation_4/PartitionedCallţ
max_pool/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_max_pool_layer_call_and_return_conditional_losses_437422
max_pool/PartitionedCallţ
average_pool/PartitionedCallPartitionedCall!max_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_average_pool_layer_call_and_return_conditional_losses_437552
average_pool/PartitionedCallŻ
dense_1/StatefulPartitionedCallStatefulPartitionedCall%average_pool/PartitionedCall:output:0dense_1_45092dense_1_45094*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_441922!
dense_1/StatefulPartitionedCallś
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_5_45097batch_normalization_5_45099batch_normalization_5_45101batch_normalization_5_45103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_437852/
-batch_normalization_5/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_442122
activation_5/PartitionedCallů
dropout_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_442192
dropout_1/PartitionedCallť
"classifier/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0classifier_45108classifier_45110*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_classifier_layer_call_and_return_conditional_losses_442322$
"classifier/StatefulPartitionedCall
IdentityIdentity+classifier/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall#^classifier/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2H
"classifier/StatefulPartitionedCall"classifier/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Z V
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
input_im
Ą

'__inference_dense_1_layer_call_fn_46548

inputs
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_441922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ż
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_44348

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46205

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ű
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs

c
G__inference_activation_1_layer_call_and_return_conditional_losses_44028

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ő
	
#__inference_cnn_layer_call_fn_45467

inputs!
unknown:		@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:
Ä>

unknown_36:	Ä>
identity˘StatefulPartitionedCallŐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*<
_read_only_resource_inputs
	
 #$%&*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_cnn_layer_call_and_return_conditional_losses_448522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ć
Î
3__inference_batch_normalization_layer_call_fn_45806

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity˘StatefulPartitionedCallľ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_431282
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

Ă
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46529

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1˙
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
łÔ
Ŕ#
 __inference__wrapped_model_43106
input_imC
)cnn_conv_1_conv2d_readvariableop_resource:		@8
*cnn_conv_1_biasadd_readvariableop_resource:@=
/cnn_batch_normalization_readvariableop_resource:@?
1cnn_batch_normalization_readvariableop_1_resource:@N
@cnn_batch_normalization_fusedbatchnormv3_readvariableop_resource:@P
Bcnn_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@D
)cnn_conv_2_conv2d_readvariableop_resource:@9
*cnn_conv_2_biasadd_readvariableop_resource:	@
1cnn_batch_normalization_1_readvariableop_resource:	B
3cnn_batch_normalization_1_readvariableop_1_resource:	Q
Bcnn_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	S
Dcnn_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	E
)cnn_conv_3_conv2d_readvariableop_resource:9
*cnn_conv_3_biasadd_readvariableop_resource:	@
1cnn_batch_normalization_2_readvariableop_resource:	B
3cnn_batch_normalization_2_readvariableop_1_resource:	Q
Bcnn_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	S
Dcnn_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	E
)cnn_conv_4_conv2d_readvariableop_resource:9
*cnn_conv_4_biasadd_readvariableop_resource:	@
1cnn_batch_normalization_3_readvariableop_resource:	B
3cnn_batch_normalization_3_readvariableop_1_resource:	Q
Bcnn_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	S
Dcnn_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	E
)cnn_conv_5_conv2d_readvariableop_resource:9
*cnn_conv_5_biasadd_readvariableop_resource:	@
1cnn_batch_normalization_4_readvariableop_resource:	B
3cnn_batch_normalization_4_readvariableop_1_resource:	Q
Bcnn_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	S
Dcnn_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	>
*cnn_dense_1_matmul_readvariableop_resource:
:
+cnn_dense_1_biasadd_readvariableop_resource:	J
;cnn_batch_normalization_5_batchnorm_readvariableop_resource:	N
?cnn_batch_normalization_5_batchnorm_mul_readvariableop_resource:	L
=cnn_batch_normalization_5_batchnorm_readvariableop_1_resource:	L
=cnn_batch_normalization_5_batchnorm_readvariableop_2_resource:	A
-cnn_classifier_matmul_readvariableop_resource:
Ä>=
.cnn_classifier_biasadd_readvariableop_resource:	Ä>
identity˘7cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp˘9cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1˘&cnn/batch_normalization/ReadVariableOp˘(cnn/batch_normalization/ReadVariableOp_1˘9cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp˘;cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1˘(cnn/batch_normalization_1/ReadVariableOp˘*cnn/batch_normalization_1/ReadVariableOp_1˘9cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp˘;cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1˘(cnn/batch_normalization_2/ReadVariableOp˘*cnn/batch_normalization_2/ReadVariableOp_1˘9cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp˘;cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1˘(cnn/batch_normalization_3/ReadVariableOp˘*cnn/batch_normalization_3/ReadVariableOp_1˘9cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp˘;cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1˘(cnn/batch_normalization_4/ReadVariableOp˘*cnn/batch_normalization_4/ReadVariableOp_1˘2cnn/batch_normalization_5/batchnorm/ReadVariableOp˘4cnn/batch_normalization_5/batchnorm/ReadVariableOp_1˘4cnn/batch_normalization_5/batchnorm/ReadVariableOp_2˘6cnn/batch_normalization_5/batchnorm/mul/ReadVariableOp˘%cnn/classifier/BiasAdd/ReadVariableOp˘$cnn/classifier/MatMul/ReadVariableOp˘!cnn/conv_1/BiasAdd/ReadVariableOp˘ cnn/conv_1/Conv2D/ReadVariableOp˘!cnn/conv_2/BiasAdd/ReadVariableOp˘ cnn/conv_2/Conv2D/ReadVariableOp˘!cnn/conv_3/BiasAdd/ReadVariableOp˘ cnn/conv_3/Conv2D/ReadVariableOp˘!cnn/conv_4/BiasAdd/ReadVariableOp˘ cnn/conv_4/Conv2D/ReadVariableOp˘!cnn/conv_5/BiasAdd/ReadVariableOp˘ cnn/conv_5/Conv2D/ReadVariableOp˘"cnn/dense_1/BiasAdd/ReadVariableOp˘!cnn/dense_1/MatMul/ReadVariableOpś
 cnn/conv_1/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:		@*
dtype02"
 cnn/conv_1/Conv2D/ReadVariableOpÇ
cnn/conv_1/Conv2DConv2Dinput_im(cnn/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*
paddingVALID*
strides
2
cnn/conv_1/Conv2D­
!cnn/conv_1/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!cnn/conv_1/BiasAdd/ReadVariableOp´
cnn/conv_1/BiasAddBiasAddcnn/conv_1/Conv2D:output:0)cnn/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2
cnn/conv_1/BiasAddź
&cnn/batch_normalization/ReadVariableOpReadVariableOp/cnn_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype02(
&cnn/batch_normalization/ReadVariableOpÂ
(cnn/batch_normalization/ReadVariableOp_1ReadVariableOp1cnn_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(cnn/batch_normalization/ReadVariableOp_1ď
7cnn/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp@cnn_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7cnn/batch_normalization/FusedBatchNormV3/ReadVariableOpő
9cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBcnn_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ď
(cnn/batch_normalization/FusedBatchNormV3FusedBatchNormV3cnn/conv_1/BiasAdd:output:0.cnn/batch_normalization/ReadVariableOp:value:00cnn/batch_normalization/ReadVariableOp_1:value:0?cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Acnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙<@:@:@:@:@:*
epsilon%o:*
is_training( 2*
(cnn/batch_normalization/FusedBatchNormV3
cnn/activation/ReluRelu,cnn/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2
cnn/activation/Reluˇ
 cnn/conv_2/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02"
 cnn/conv_2/Conv2D/ReadVariableOpŕ
cnn/conv_2/Conv2DConv2D!cnn/activation/Relu:activations:0(cnn/conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
cnn/conv_2/Conv2DŽ
!cnn/conv_2/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!cnn/conv_2/BiasAdd/ReadVariableOpľ
cnn/conv_2/BiasAddBiasAddcnn/conv_2/Conv2D:output:0)cnn/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
cnn/conv_2/BiasAddĂ
(cnn/batch_normalization_1/ReadVariableOpReadVariableOp1cnn_batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype02*
(cnn/batch_normalization_1/ReadVariableOpÉ
*cnn/batch_normalization_1/ReadVariableOp_1ReadVariableOp3cnn_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype02,
*cnn/batch_normalization_1/ReadVariableOp_1ö
9cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBcnn_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02;
9cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOpü
;cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDcnn_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02=
;cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1
*cnn/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3cnn/conv_2/BiasAdd:output:00cnn/batch_normalization_1/ReadVariableOp:value:02cnn/batch_normalization_1/ReadVariableOp_1:value:0Acnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Ccnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2,
*cnn/batch_normalization_1/FusedBatchNormV3Ą
cnn/activation_1/ReluRelu.cnn/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
cnn/activation_1/Relu¸
 cnn/conv_3/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 cnn/conv_3/Conv2D/ReadVariableOpâ
cnn/conv_3/Conv2DConv2D#cnn/activation_1/Relu:activations:0(cnn/conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
cnn/conv_3/Conv2DŽ
!cnn/conv_3/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!cnn/conv_3/BiasAdd/ReadVariableOpľ
cnn/conv_3/BiasAddBiasAddcnn/conv_3/Conv2D:output:0)cnn/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
cnn/conv_3/BiasAddĂ
(cnn/batch_normalization_2/ReadVariableOpReadVariableOp1cnn_batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype02*
(cnn/batch_normalization_2/ReadVariableOpÉ
*cnn/batch_normalization_2/ReadVariableOp_1ReadVariableOp3cnn_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype02,
*cnn/batch_normalization_2/ReadVariableOp_1ö
9cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBcnn_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02;
9cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOpü
;cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDcnn_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02=
;cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1
*cnn/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3cnn/conv_3/BiasAdd:output:00cnn/batch_normalization_2/ReadVariableOp:value:02cnn/batch_normalization_2/ReadVariableOp_1:value:0Acnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Ccnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2,
*cnn/batch_normalization_2/FusedBatchNormV3Ą
cnn/activation_2/ReluRelu.cnn/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
cnn/activation_2/Relu¸
 cnn/conv_4/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 cnn/conv_4/Conv2D/ReadVariableOpâ
cnn/conv_4/Conv2DConv2D#cnn/activation_2/Relu:activations:0(cnn/conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
cnn/conv_4/Conv2DŽ
!cnn/conv_4/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!cnn/conv_4/BiasAdd/ReadVariableOpľ
cnn/conv_4/BiasAddBiasAddcnn/conv_4/Conv2D:output:0)cnn/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
cnn/conv_4/BiasAddĂ
(cnn/batch_normalization_3/ReadVariableOpReadVariableOp1cnn_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype02*
(cnn/batch_normalization_3/ReadVariableOpÉ
*cnn/batch_normalization_3/ReadVariableOp_1ReadVariableOp3cnn_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02,
*cnn/batch_normalization_3/ReadVariableOp_1ö
9cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpBcnn_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02;
9cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOpü
;cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDcnn_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02=
;cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1
*cnn/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3cnn/conv_4/BiasAdd:output:00cnn/batch_normalization_3/ReadVariableOp:value:02cnn/batch_normalization_3/ReadVariableOp_1:value:0Acnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ccnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2,
*cnn/batch_normalization_3/FusedBatchNormV3Ą
cnn/activation_3/ReluRelu.cnn/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
cnn/activation_3/Relu¸
 cnn/conv_5/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 cnn/conv_5/Conv2D/ReadVariableOpâ
cnn/conv_5/Conv2DConv2D#cnn/activation_3/Relu:activations:0(cnn/conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
paddingSAME*
strides
2
cnn/conv_5/Conv2DŽ
!cnn/conv_5/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!cnn/conv_5/BiasAdd/ReadVariableOpľ
cnn/conv_5/BiasAddBiasAddcnn/conv_5/Conv2D:output:0)cnn/conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
cnn/conv_5/BiasAddĂ
(cnn/batch_normalization_4/ReadVariableOpReadVariableOp1cnn_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02*
(cnn/batch_normalization_4/ReadVariableOpÉ
*cnn/batch_normalization_4/ReadVariableOp_1ReadVariableOp3cnn_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02,
*cnn/batch_normalization_4/ReadVariableOp_1ö
9cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpBcnn_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02;
9cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOpü
;cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDcnn_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02=
;cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1
*cnn/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3cnn/conv_5/BiasAdd:output:00cnn/batch_normalization_4/ReadVariableOp:value:02cnn/batch_normalization_4/ReadVariableOp_1:value:0Acnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Ccnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2,
*cnn/batch_normalization_4/FusedBatchNormV3Ą
cnn/activation_4/ReluRelu.cnn/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
cnn/activation_4/ReluĘ
cnn/max_pool/MaxPoolMaxPool#cnn/activation_4/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
cnn/max_pool/MaxPoolŁ
'cnn/average_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2)
'cnn/average_pool/Mean/reduction_indicesş
cnn/average_pool/MeanMeancnn/max_pool/MaxPool:output:00cnn/average_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cnn/average_pool/Meanł
!cnn/dense_1/MatMul/ReadVariableOpReadVariableOp*cnn_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!cnn/dense_1/MatMul/ReadVariableOp°
cnn/dense_1/MatMulMatMulcnn/average_pool/Mean:output:0)cnn/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cnn/dense_1/MatMulą
"cnn/dense_1/BiasAdd/ReadVariableOpReadVariableOp+cnn_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"cnn/dense_1/BiasAdd/ReadVariableOp˛
cnn/dense_1/BiasAddBiasAddcnn/dense_1/MatMul:product:0*cnn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cnn/dense_1/BiasAddá
2cnn/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp;cnn_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype024
2cnn/batch_normalization_5/batchnorm/ReadVariableOp
)cnn/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)cnn/batch_normalization_5/batchnorm/add/yń
'cnn/batch_normalization_5/batchnorm/addAddV2:cnn/batch_normalization_5/batchnorm/ReadVariableOp:value:02cnn/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2)
'cnn/batch_normalization_5/batchnorm/add˛
)cnn/batch_normalization_5/batchnorm/RsqrtRsqrt+cnn/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:2+
)cnn/batch_normalization_5/batchnorm/Rsqrtí
6cnn/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp?cnn_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype028
6cnn/batch_normalization_5/batchnorm/mul/ReadVariableOpî
'cnn/batch_normalization_5/batchnorm/mulMul-cnn/batch_normalization_5/batchnorm/Rsqrt:y:0>cnn/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2)
'cnn/batch_normalization_5/batchnorm/mulŰ
)cnn/batch_normalization_5/batchnorm/mul_1Mulcnn/dense_1/BiasAdd:output:0+cnn/batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)cnn/batch_normalization_5/batchnorm/mul_1ç
4cnn/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp=cnn_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4cnn/batch_normalization_5/batchnorm/ReadVariableOp_1î
)cnn/batch_normalization_5/batchnorm/mul_2Mul<cnn/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0+cnn/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:2+
)cnn/batch_normalization_5/batchnorm/mul_2ç
4cnn/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp=cnn_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype026
4cnn/batch_normalization_5/batchnorm/ReadVariableOp_2ě
'cnn/batch_normalization_5/batchnorm/subSub<cnn/batch_normalization_5/batchnorm/ReadVariableOp_2:value:0-cnn/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2)
'cnn/batch_normalization_5/batchnorm/subî
)cnn/batch_normalization_5/batchnorm/add_1AddV2-cnn/batch_normalization_5/batchnorm/mul_1:z:0+cnn/batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)cnn/batch_normalization_5/batchnorm/add_1
cnn/activation_5/ReluRelu-cnn/batch_normalization_5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cnn/activation_5/Relu
cnn/dropout_1/IdentityIdentity#cnn/activation_5/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cnn/dropout_1/Identityź
$cnn/classifier/MatMul/ReadVariableOpReadVariableOp-cnn_classifier_matmul_readvariableop_resource* 
_output_shapes
:
Ä>*
dtype02&
$cnn/classifier/MatMul/ReadVariableOpş
cnn/classifier/MatMulMatMulcnn/dropout_1/Identity:output:0,cnn/classifier/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
cnn/classifier/MatMulş
%cnn/classifier/BiasAdd/ReadVariableOpReadVariableOp.cnn_classifier_biasadd_readvariableop_resource*
_output_shapes	
:Ä>*
dtype02'
%cnn/classifier/BiasAdd/ReadVariableOpž
cnn/classifier/BiasAddBiasAddcnn/classifier/MatMul:product:0-cnn/classifier/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
cnn/classifier/BiasAdd
cnn/classifier/SoftmaxSoftmaxcnn/classifier/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2
cnn/classifier/SoftmaxŢ
IdentityIdentity cnn/classifier/Softmax:softmax:08^cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp:^cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1'^cnn/batch_normalization/ReadVariableOp)^cnn/batch_normalization/ReadVariableOp_1:^cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp<^cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1)^cnn/batch_normalization_1/ReadVariableOp+^cnn/batch_normalization_1/ReadVariableOp_1:^cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp<^cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1)^cnn/batch_normalization_2/ReadVariableOp+^cnn/batch_normalization_2/ReadVariableOp_1:^cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp<^cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1)^cnn/batch_normalization_3/ReadVariableOp+^cnn/batch_normalization_3/ReadVariableOp_1:^cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp<^cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1)^cnn/batch_normalization_4/ReadVariableOp+^cnn/batch_normalization_4/ReadVariableOp_13^cnn/batch_normalization_5/batchnorm/ReadVariableOp5^cnn/batch_normalization_5/batchnorm/ReadVariableOp_15^cnn/batch_normalization_5/batchnorm/ReadVariableOp_27^cnn/batch_normalization_5/batchnorm/mul/ReadVariableOp&^cnn/classifier/BiasAdd/ReadVariableOp%^cnn/classifier/MatMul/ReadVariableOp"^cnn/conv_1/BiasAdd/ReadVariableOp!^cnn/conv_1/Conv2D/ReadVariableOp"^cnn/conv_2/BiasAdd/ReadVariableOp!^cnn/conv_2/Conv2D/ReadVariableOp"^cnn/conv_3/BiasAdd/ReadVariableOp!^cnn/conv_3/Conv2D/ReadVariableOp"^cnn/conv_4/BiasAdd/ReadVariableOp!^cnn/conv_4/Conv2D/ReadVariableOp"^cnn/conv_5/BiasAdd/ReadVariableOp!^cnn/conv_5/Conv2D/ReadVariableOp#^cnn/dense_1/BiasAdd/ReadVariableOp"^cnn/dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp7cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp2v
9cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_19cnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_12P
&cnn/batch_normalization/ReadVariableOp&cnn/batch_normalization/ReadVariableOp2T
(cnn/batch_normalization/ReadVariableOp_1(cnn/batch_normalization/ReadVariableOp_12v
9cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp9cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2z
;cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1;cnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12T
(cnn/batch_normalization_1/ReadVariableOp(cnn/batch_normalization_1/ReadVariableOp2X
*cnn/batch_normalization_1/ReadVariableOp_1*cnn/batch_normalization_1/ReadVariableOp_12v
9cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp9cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2z
;cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1;cnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12T
(cnn/batch_normalization_2/ReadVariableOp(cnn/batch_normalization_2/ReadVariableOp2X
*cnn/batch_normalization_2/ReadVariableOp_1*cnn/batch_normalization_2/ReadVariableOp_12v
9cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp9cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2z
;cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;cnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12T
(cnn/batch_normalization_3/ReadVariableOp(cnn/batch_normalization_3/ReadVariableOp2X
*cnn/batch_normalization_3/ReadVariableOp_1*cnn/batch_normalization_3/ReadVariableOp_12v
9cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp9cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2z
;cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;cnn/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12T
(cnn/batch_normalization_4/ReadVariableOp(cnn/batch_normalization_4/ReadVariableOp2X
*cnn/batch_normalization_4/ReadVariableOp_1*cnn/batch_normalization_4/ReadVariableOp_12h
2cnn/batch_normalization_5/batchnorm/ReadVariableOp2cnn/batch_normalization_5/batchnorm/ReadVariableOp2l
4cnn/batch_normalization_5/batchnorm/ReadVariableOp_14cnn/batch_normalization_5/batchnorm/ReadVariableOp_12l
4cnn/batch_normalization_5/batchnorm/ReadVariableOp_24cnn/batch_normalization_5/batchnorm/ReadVariableOp_22p
6cnn/batch_normalization_5/batchnorm/mul/ReadVariableOp6cnn/batch_normalization_5/batchnorm/mul/ReadVariableOp2N
%cnn/classifier/BiasAdd/ReadVariableOp%cnn/classifier/BiasAdd/ReadVariableOp2L
$cnn/classifier/MatMul/ReadVariableOp$cnn/classifier/MatMul/ReadVariableOp2F
!cnn/conv_1/BiasAdd/ReadVariableOp!cnn/conv_1/BiasAdd/ReadVariableOp2D
 cnn/conv_1/Conv2D/ReadVariableOp cnn/conv_1/Conv2D/ReadVariableOp2F
!cnn/conv_2/BiasAdd/ReadVariableOp!cnn/conv_2/BiasAdd/ReadVariableOp2D
 cnn/conv_2/Conv2D/ReadVariableOp cnn/conv_2/Conv2D/ReadVariableOp2F
!cnn/conv_3/BiasAdd/ReadVariableOp!cnn/conv_3/BiasAdd/ReadVariableOp2D
 cnn/conv_3/Conv2D/ReadVariableOp cnn/conv_3/Conv2D/ReadVariableOp2F
!cnn/conv_4/BiasAdd/ReadVariableOp!cnn/conv_4/BiasAdd/ReadVariableOp2D
 cnn/conv_4/Conv2D/ReadVariableOp cnn/conv_4/Conv2D/ReadVariableOp2F
!cnn/conv_5/BiasAdd/ReadVariableOp!cnn/conv_5/BiasAdd/ReadVariableOp2D
 cnn/conv_5/Conv2D/ReadVariableOp cnn/conv_5/Conv2D/ReadVariableOp2H
"cnn/dense_1/BiasAdd/ReadVariableOp"cnn/dense_1/BiasAdd/ReadVariableOp2F
!cnn/dense_1/MatMul/ReadVariableOp!cnn/dense_1/MatMul/ReadVariableOp:Z V
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
input_im
ő
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_44219

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44013

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ű
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
¨
Ô
5__inference_batch_normalization_2_layer_call_fn_46151

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_445262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ż
ł
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46604

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity˘batchnorm/ReadVariableOp˘batchnorm/ReadVariableOp_1˘batchnorm/ReadVariableOp_2˘batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ş
Ô
5__inference_batch_normalization_1_layer_call_fn_45985

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_440132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ň

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44063

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ű
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs

Ă
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_44586

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1˙
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
ă
c
G__inference_activation_5_layer_call_and_return_conditional_losses_46648

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ă
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_44526

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1˙
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ň

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46052

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ű
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
đ
Ô
5__inference_batch_normalization_2_layer_call_fn_46125

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_434242
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ŕ

N__inference_batch_normalization_layer_call_and_return_conditional_losses_43963

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙<@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ú
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙<@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs
*
í
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43845

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity˘AssignMovingAvg˘AssignMovingAvg/ReadVariableOp˘AssignMovingAvg_1˘ AssignMovingAvg_1/ReadVariableOp˘batchnorm/ReadVariableOp˘batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradientĽ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indicesł
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decayĽ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mulż
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayŤ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpĄ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ă
E
)__inference_dropout_1_layer_call_fn_46653

inputs
identityĆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_442192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


N__inference_batch_normalization_layer_call_and_return_conditional_losses_43128

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ě
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

Ô
5__inference_batch_normalization_5_layer_call_fn_46584

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_438452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
É
H
,__inference_activation_5_layer_call_fn_46643

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_442122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_44113

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ű
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46016

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3í
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46358

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ď
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ű
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs

c
G__inference_activation_2_layer_call_and_return_conditional_losses_44078

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
é
H
,__inference_activation_4_layer_call_fn_46534

inputs
identityŃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_441782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
Ď
b
)__inference_dropout_1_layer_call_fn_46658

inputs
identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_443482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
o
â
>__inference_cnn_layer_call_and_return_conditional_losses_44239

inputs&
conv_1_43941:		@
conv_1_43943:@'
batch_normalization_43964:@'
batch_normalization_43966:@'
batch_normalization_43968:@'
batch_normalization_43970:@'
conv_2_43991:@
conv_2_43993:	*
batch_normalization_1_44014:	*
batch_normalization_1_44016:	*
batch_normalization_1_44018:	*
batch_normalization_1_44020:	(
conv_3_44041:
conv_3_44043:	*
batch_normalization_2_44064:	*
batch_normalization_2_44066:	*
batch_normalization_2_44068:	*
batch_normalization_2_44070:	(
conv_4_44091:
conv_4_44093:	*
batch_normalization_3_44114:	*
batch_normalization_3_44116:	*
batch_normalization_3_44118:	*
batch_normalization_3_44120:	(
conv_5_44141:
conv_5_44143:	*
batch_normalization_4_44164:	*
batch_normalization_4_44166:	*
batch_normalization_4_44168:	*
batch_normalization_4_44170:	!
dense_1_44193:

dense_1_44195:	*
batch_normalization_5_44198:	*
batch_normalization_5_44200:	*
batch_normalization_5_44202:	*
batch_normalization_5_44204:	$
classifier_44233:
Ä>
classifier_44235:	Ä>
identity˘+batch_normalization/StatefulPartitionedCall˘-batch_normalization_1/StatefulPartitionedCall˘-batch_normalization_2/StatefulPartitionedCall˘-batch_normalization_3/StatefulPartitionedCall˘-batch_normalization_4/StatefulPartitionedCall˘-batch_normalization_5/StatefulPartitionedCall˘"classifier/StatefulPartitionedCall˘conv_1/StatefulPartitionedCall˘conv_2/StatefulPartitionedCall˘conv_3/StatefulPartitionedCall˘conv_4/StatefulPartitionedCall˘conv_5/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_43941conv_1_43943*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_439402 
conv_1/StatefulPartitionedCallŽ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_43964batch_normalization_43966batch_normalization_43968batch_normalization_43970*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_439632-
+batch_normalization/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_439782
activation/PartitionedCall°
conv_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv_2_43991conv_2_43993*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_439902 
conv_2/StatefulPartitionedCall˝
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_44014batch_normalization_1_44016batch_normalization_1_44018batch_normalization_1_44020*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_440132/
-batch_normalization_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_440282
activation_1/PartitionedCall˛
conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv_3_44041conv_3_44043*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_440402 
conv_3/StatefulPartitionedCall˝
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_44064batch_normalization_2_44066batch_normalization_2_44068batch_normalization_2_44070*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_440632/
-batch_normalization_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_440782
activation_2/PartitionedCall˛
conv_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv_4_44091conv_4_44093*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_440902 
conv_4/StatefulPartitionedCall˝
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_44114batch_normalization_3_44116batch_normalization_3_44118batch_normalization_3_44120*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_441132/
-batch_normalization_3/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_441282
activation_3/PartitionedCall˛
conv_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv_5_44141conv_5_44143*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_441402 
conv_5/StatefulPartitionedCall˝
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_44164batch_normalization_4_44166batch_normalization_4_44168batch_normalization_4_44170*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_441632/
-batch_normalization_4/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_441782
activation_4/PartitionedCallţ
max_pool/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_max_pool_layer_call_and_return_conditional_losses_437422
max_pool/PartitionedCallţ
average_pool/PartitionedCallPartitionedCall!max_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_average_pool_layer_call_and_return_conditional_losses_437552
average_pool/PartitionedCallŻ
dense_1/StatefulPartitionedCallStatefulPartitionedCall%average_pool/PartitionedCall:output:0dense_1_44193dense_1_44195*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_441922!
dense_1/StatefulPartitionedCallś
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_5_44198batch_normalization_5_44200batch_normalization_5_44202batch_normalization_5_44204*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_437852/
-batch_normalization_5/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_442122
activation_5/PartitionedCallů
dropout_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_442192
dropout_1/PartitionedCallť
"classifier/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0classifier_44233classifier_44235*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_classifier_layer_call_and_return_conditional_losses_442322$
"classifier/StatefulPartitionedCall
IdentityIdentity+classifier/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall#^classifier/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:˙˙˙˙˙˙˙˙˙ : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2H
"classifier/StatefulPartitionedCall"classifier/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ş
Ô
5__inference_batch_normalization_4_layer_call_fn_46444

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_441632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
*
í
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46638

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity˘AssignMovingAvg˘AssignMovingAvg/ReadVariableOp˘AssignMovingAvg_1˘ AssignMovingAvg_1/ReadVariableOp˘batchnorm/ReadVariableOp˘batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradientĽ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indicesł
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decayĽ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mulż
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayŤ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpĄ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ă
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46223

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1˙
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
ő
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_46663

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Î
3__inference_batch_normalization_layer_call_fn_45832

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity˘StatefulPartitionedCallŁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙<@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_439632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙<@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs


P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46475

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3í
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ă
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46070

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1˙
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
é
H
,__inference_activation_1_layer_call_fn_46075

inputs
identityŃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_440282
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙<:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs
đ
Ô
5__inference_batch_normalization_1_layer_call_fn_45972

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_432982
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ä

&__inference_conv_2_layer_call_fn_45936

inputs"
unknown:@
	unknown_0:	
identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_439902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙<@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙<@
 
_user_specified_nameinputs
Ł
_
C__inference_max_pool_layer_call_and_return_conditional_losses_43742

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
§

*__inference_classifier_layer_call_fn_46684

inputs
unknown:
Ä>
	unknown_0:	Ä>
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_classifier_layer_call_and_return_conditional_losses_442322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ă
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44406

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity˘AssignNewValue˘AssignNewValue_1˘FusedBatchNormV3/ReadVariableOp˘!FusedBatchNormV3/ReadVariableOp_1˘ReadVariableOp˘ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpŽ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙<:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1˙
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙<: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙<
 
_user_specified_nameinputs"ĚL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*š
serving_defaultĽ
F
input_im:
serving_default_input_im:0˙˙˙˙˙˙˙˙˙ ?

classifier1
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙Ä>tensorflow/serving/predict:§
Ć
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Ů_default_save_signature
Ú__call__
+Ű&call_and_return_all_conditional_losses"˝ż
_tf_keras_network ż{"name": "cnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "cnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_im"}, "name": "input_im", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_im", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "average_pool", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "average_pool", "inbound_nodes": [[["max_pool", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["average_pool", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "classifier", "trainable": true, "dtype": "float32", "units": 8004, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "classifier", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": [["input_im", 0, 0]], "output_layers": [["classifier", 0, 0]]}, "shared_object_id": 61, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 128, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 128, 1]}, "float32", "input_im"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "cnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_im"}, "name": "input_im", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_im", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["activation", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["activation_1", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["activation_2", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 32}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 34}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv_4", 0, 0, {}]]], "shared_object_id": 35}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["activation_3", 0, 0, {}]]], "shared_object_id": 39}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 41}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 43}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]], "shared_object_id": 45}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool", "inbound_nodes": [[["activation_4", 0, 0, {}]]], "shared_object_id": 46}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "average_pool", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "average_pool", "inbound_nodes": [[["max_pool", 0, 0, {}]]], "shared_object_id": 47}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["average_pool", 0, 0, {}]]], "shared_object_id": 50}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 51}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 52}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 54}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 55}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]], "shared_object_id": 56}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["activation_5", 0, 0, {}]]], "shared_object_id": 57}, {"class_name": "Dense", "config": {"name": "classifier", "trainable": true, "dtype": "float32", "units": 8004, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "classifier", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 60}], "input_layers": [["input_im", 0, 0]], "output_layers": [["classifier", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 63}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 1.0000000656873453e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ý"ú
_tf_keras_input_layerÚ{"class_name": "InputLayer", "name": "input_im", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_im"}}
ü


kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"Ő	
_tf_keras_layerť	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_im", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 128, 1]}}
î

$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*trainable_variables
+	variables
,	keras_api
Ţ__call__
+ß&call_and_return_all_conditional_losses"	
_tf_keras_layerţ{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 64]}}
˘
-regularization_losses
.trainable_variables
/	variables
0	keras_api
ŕ__call__
+á&call_and_return_all_conditional_losses"
_tf_keras_layer÷{"name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization", 0, 0, {}]]], "shared_object_id": 9}


1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
â__call__
+ă&call_and_return_all_conditional_losses"Ű	
_tf_keras_layerÁ	{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 64]}}
ů

7axis
	8gamma
9beta
:moving_mean
;moving_variance
<regularization_losses
=trainable_variables
>	variables
?	keras_api
ä__call__
+ĺ&call_and_return_all_conditional_losses"Ł	
_tf_keras_layer	{"name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 128]}}
Š
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
ć__call__
+ç&call_and_return_all_conditional_losses"
_tf_keras_layerţ{"name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]], "shared_object_id": 18}


Dkernel
Ebias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
č__call__
+é&call_and_return_all_conditional_losses"ß	
_tf_keras_layerĹ	{"name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation_1", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 128]}}
ů

Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
ę__call__
+ë&call_and_return_all_conditional_losses"Ł	
_tf_keras_layer	{"name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 128]}}
Š
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
ě__call__
+í&call_and_return_all_conditional_losses"
_tf_keras_layerţ{"name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]], "shared_object_id": 27}


Wkernel
Xbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
î__call__
+ď&call_and_return_all_conditional_losses"ß	
_tf_keras_layerĹ	{"name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation_2", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 128]}}
ů

]axis
	^gamma
_beta
`moving_mean
amoving_variance
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
đ__call__
+ń&call_and_return_all_conditional_losses"Ł	
_tf_keras_layer	{"name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 32}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 34}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv_4", 0, 0, {}]]], "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 256]}}
Š
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
ň__call__
+ó&call_and_return_all_conditional_losses"
_tf_keras_layerţ{"name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]], "shared_object_id": 36}


jkernel
kbias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
ô__call__
+ő&call_and_return_all_conditional_losses"ß	
_tf_keras_layerĹ	{"name": "conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation_3", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 256]}}
ů

paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"Ł	
_tf_keras_layer	{"name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 41}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 43}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 60, 256]}}
Š
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
ř__call__
+ů&call_and_return_all_conditional_losses"
_tf_keras_layerţ{"name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]], "shared_object_id": 45}
Ő
}regularization_losses
~trainable_variables
	variables
	keras_api
ú__call__
+ű&call_and_return_all_conditional_losses"Ă
_tf_keras_layerŠ{"name": "max_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["activation_4", 0, 0, {}]]], "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 74}}
Ţ
regularization_losses
trainable_variables
	variables
	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"É
_tf_keras_layerŻ{"name": "average_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalAveragePooling2D", "config": {"name": "average_pool", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["max_pool", 0, 0, {}]]], "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 75}}
	
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
ţ__call__
+˙&call_and_return_all_conditional_losses"á
_tf_keras_layerÇ{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["average_pool", 0, 0, {}]]], "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ű

	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer	{"name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 51}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 52}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 54}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
­
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerţ{"name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]], "shared_object_id": 56}
´
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "inbound_nodes": [[["activation_5", 0, 0, {}]]], "shared_object_id": 57}
	
kernel
	bias
regularization_losses
trainable_variables
 	variables
Ą	keras_api
__call__
+&call_and_return_all_conditional_losses"ć
_tf_keras_layerĚ{"name": "classifier", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "classifier", "trainable": true, "dtype": "float32", "units": 8004, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ě
	˘iter
Łbeta_1
¤beta_2

Ľdecay
Ślearning_ratemĽmŚ%m§&m¨1mŠ2mŞ8mŤ9mŹDm­EmŽKmŻLm°WmąXm˛^mł_m´jmľkmśqmˇrm¸	mš	mş	mť	mź	m˝	mžvżvŔ%vÁ&vÂ1vĂ2vÄ8vĹ9vĆDvÇEvČKvÉLvĘWvËXvĚ^vÍ_vÎjvĎkvĐqvŃrvŇ	vÓ	vÔ	vŐ	vÖ	v×	vŘ"
	optimizer
 "
trackable_list_wrapper
ě
0
1
%2
&3
14
25
86
97
D8
E9
K10
L11
W12
X13
^14
_15
j16
k17
q18
r19
20
21
22
23
24
25"
trackable_list_wrapper
Î
0
1
%2
&3
'4
(5
16
27
88
99
:10
;11
D12
E13
K14
L15
M16
N17
W18
X19
^20
_21
`22
a23
j24
k25
q26
r27
s28
t29
30
31
32
33
34
35
36
37"
trackable_list_wrapper
Ó
§non_trainable_variables
 ¨layer_regularization_losses
Šlayers
Şlayer_metrics
Ťmetrics
regularization_losses
trainable_variables
	variables
Ú__call__
Ů_default_save_signature
+Ű&call_and_return_all_conditional_losses
'Ű"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
':%		@2conv_1/kernel
:@2conv_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ľ
Źnon_trainable_variables
 ­layer_regularization_losses
Žlayers
Żlayer_metrics
°metrics
 regularization_losses
!trainable_variables
"	variables
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
ľ
ąnon_trainable_variables
 ˛layer_regularization_losses
łlayers
´layer_metrics
ľmetrics
)regularization_losses
*trainable_variables
+	variables
Ţ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
śnon_trainable_variables
 ˇlayer_regularization_losses
¸layers
šlayer_metrics
şmetrics
-regularization_losses
.trainable_variables
/	variables
ŕ__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
(:&@2conv_2/kernel
:2conv_2/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
ľ
ťnon_trainable_variables
 źlayer_regularization_losses
˝layers
žlayer_metrics
żmetrics
3regularization_losses
4trainable_variables
5	variables
â__call__
+ă&call_and_return_all_conditional_losses
'ă"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_1/gamma
):'2batch_normalization_1/beta
2:0 (2!batch_normalization_1/moving_mean
6:4 (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
ľ
Ŕnon_trainable_variables
 Álayer_regularization_losses
Âlayers
Ălayer_metrics
Ämetrics
<regularization_losses
=trainable_variables
>	variables
ä__call__
+ĺ&call_and_return_all_conditional_losses
'ĺ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Ĺnon_trainable_variables
 Ćlayer_regularization_losses
Çlayers
Člayer_metrics
Émetrics
@regularization_losses
Atrainable_variables
B	variables
ć__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
):'2conv_3/kernel
:2conv_3/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
ľ
Ęnon_trainable_variables
 Ëlayer_regularization_losses
Ělayers
Ílayer_metrics
Îmetrics
Fregularization_losses
Gtrainable_variables
H	variables
č__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_2/gamma
):'2batch_normalization_2/beta
2:0 (2!batch_normalization_2/moving_mean
6:4 (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
<
K0
L1
M2
N3"
trackable_list_wrapper
ľ
Ďnon_trainable_variables
 Đlayer_regularization_losses
Ńlayers
Ňlayer_metrics
Ómetrics
Oregularization_losses
Ptrainable_variables
Q	variables
ę__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Ônon_trainable_variables
 Őlayer_regularization_losses
Ölayers
×layer_metrics
Řmetrics
Sregularization_losses
Ttrainable_variables
U	variables
ě__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
):'2conv_4/kernel
:2conv_4/bias
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
ľ
Ůnon_trainable_variables
 Úlayer_regularization_losses
Űlayers
Ülayer_metrics
Ýmetrics
Yregularization_losses
Ztrainable_variables
[	variables
î__call__
+ď&call_and_return_all_conditional_losses
'ď"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_3/gamma
):'2batch_normalization_3/beta
2:0 (2!batch_normalization_3/moving_mean
6:4 (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
ľ
Ţnon_trainable_variables
 ßlayer_regularization_losses
ŕlayers
álayer_metrics
âmetrics
bregularization_losses
ctrainable_variables
d	variables
đ__call__
+ń&call_and_return_all_conditional_losses
'ń"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
ănon_trainable_variables
 älayer_regularization_losses
ĺlayers
ćlayer_metrics
çmetrics
fregularization_losses
gtrainable_variables
h	variables
ň__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
):'2conv_5/kernel
:2conv_5/bias
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
ľ
čnon_trainable_variables
 élayer_regularization_losses
ęlayers
ëlayer_metrics
ěmetrics
lregularization_losses
mtrainable_variables
n	variables
ô__call__
+ő&call_and_return_all_conditional_losses
'ő"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_4/gamma
):'2batch_normalization_4/beta
2:0 (2!batch_normalization_4/moving_mean
6:4 (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
<
q0
r1
s2
t3"
trackable_list_wrapper
ľ
ínon_trainable_variables
 îlayer_regularization_losses
ďlayers
đlayer_metrics
ńmetrics
uregularization_losses
vtrainable_variables
w	variables
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
ňnon_trainable_variables
 ólayer_regularization_losses
ôlayers
őlayer_metrics
ömetrics
yregularization_losses
ztrainable_variables
{	variables
ř__call__
+ů&call_and_return_all_conditional_losses
'ů"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
÷non_trainable_variables
 řlayer_regularization_losses
ůlayers
úlayer_metrics
űmetrics
}regularization_losses
~trainable_variables
	variables
ú__call__
+ű&call_and_return_all_conditional_losses
'ű"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ünon_trainable_variables
 ýlayer_regularization_losses
ţlayers
˙layer_metrics
metrics
regularization_losses
trainable_variables
	variables
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
	variables
ţ__call__
+˙&call_and_return_all_conditional_losses
'˙"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_5/gamma
):'2batch_normalization_5/beta
2:0 (2!batch_normalization_5/moving_mean
6:4 (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
¸
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#
Ä>2classifier/kernel
:Ä>2classifier/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
non_trainable_variables
 layer_regularization_losses
layers
layer_metrics
metrics
regularization_losses
trainable_variables
 	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
x
'0
(1
:2
;3
M4
N5
`6
a7
s8
t9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Ř

total

count
	variables
	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 79}
Ł

 total

Ącount
˘
_fn_kwargs
Ł	variables
¤	keras_api"×
_tf_keras_metricź{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 63}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
 0
Ą1"
trackable_list_wrapper
.
Ł	variables"
_generic_user_object
,:*		@2Adam/conv_1/kernel/m
:@2Adam/conv_1/bias/m
,:*@2 Adam/batch_normalization/gamma/m
+:)@2Adam/batch_normalization/beta/m
-:+@2Adam/conv_2/kernel/m
:2Adam/conv_2/bias/m
/:-2"Adam/batch_normalization_1/gamma/m
.:,2!Adam/batch_normalization_1/beta/m
.:,2Adam/conv_3/kernel/m
:2Adam/conv_3/bias/m
/:-2"Adam/batch_normalization_2/gamma/m
.:,2!Adam/batch_normalization_2/beta/m
.:,2Adam/conv_4/kernel/m
:2Adam/conv_4/bias/m
/:-2"Adam/batch_normalization_3/gamma/m
.:,2!Adam/batch_normalization_3/beta/m
.:,2Adam/conv_5/kernel/m
:2Adam/conv_5/bias/m
/:-2"Adam/batch_normalization_4/gamma/m
.:,2!Adam/batch_normalization_4/beta/m
':%
2Adam/dense_1/kernel/m
 :2Adam/dense_1/bias/m
/:-2"Adam/batch_normalization_5/gamma/m
.:,2!Adam/batch_normalization_5/beta/m
*:(
Ä>2Adam/classifier/kernel/m
#:!Ä>2Adam/classifier/bias/m
,:*		@2Adam/conv_1/kernel/v
:@2Adam/conv_1/bias/v
,:*@2 Adam/batch_normalization/gamma/v
+:)@2Adam/batch_normalization/beta/v
-:+@2Adam/conv_2/kernel/v
:2Adam/conv_2/bias/v
/:-2"Adam/batch_normalization_1/gamma/v
.:,2!Adam/batch_normalization_1/beta/v
.:,2Adam/conv_3/kernel/v
:2Adam/conv_3/bias/v
/:-2"Adam/batch_normalization_2/gamma/v
.:,2!Adam/batch_normalization_2/beta/v
.:,2Adam/conv_4/kernel/v
:2Adam/conv_4/bias/v
/:-2"Adam/batch_normalization_3/gamma/v
.:,2!Adam/batch_normalization_3/beta/v
.:,2Adam/conv_5/kernel/v
:2Adam/conv_5/bias/v
/:-2"Adam/batch_normalization_4/gamma/v
.:,2!Adam/batch_normalization_4/beta/v
':%
2Adam/dense_1/kernel/v
 :2Adam/dense_1/bias/v
/:-2"Adam/batch_normalization_5/gamma/v
.:,2!Adam/batch_normalization_5/beta/v
*:(
Ä>2Adam/classifier/kernel/v
#:!Ä>2Adam/classifier/bias/v
č2ĺ
 __inference__wrapped_model_43106Ŕ
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *0˘-
+(
input_im˙˙˙˙˙˙˙˙˙ 
Ú2×
#__inference_cnn_layer_call_fn_44318
#__inference_cnn_layer_call_fn_45386
#__inference_cnn_layer_call_fn_45467
#__inference_cnn_layer_call_fn_45012Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ć2Ă
>__inference_cnn_layer_call_and_return_conditional_losses_45610
>__inference_cnn_layer_call_and_return_conditional_losses_45774
>__inference_cnn_layer_call_and_return_conditional_losses_45114
>__inference_cnn_layer_call_and_return_conditional_losses_45216Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Đ2Í
&__inference_conv_1_layer_call_fn_45783˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_conv_1_layer_call_and_return_conditional_losses_45793˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
3__inference_batch_normalization_layer_call_fn_45806
3__inference_batch_normalization_layer_call_fn_45819
3__inference_batch_normalization_layer_call_fn_45832
3__inference_batch_normalization_layer_call_fn_45845´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ú2÷
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45863
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45881
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45899
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45917´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ô2Ń
*__inference_activation_layer_call_fn_45922˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ď2ě
E__inference_activation_layer_call_and_return_conditional_losses_45927˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_conv_2_layer_call_fn_45936˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_conv_2_layer_call_and_return_conditional_losses_45946˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
5__inference_batch_normalization_1_layer_call_fn_45959
5__inference_batch_normalization_1_layer_call_fn_45972
5__inference_batch_normalization_1_layer_call_fn_45985
5__inference_batch_normalization_1_layer_call_fn_45998´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
2˙
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46016
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46034
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46052
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46070´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
,__inference_activation_1_layer_call_fn_46075˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
G__inference_activation_1_layer_call_and_return_conditional_losses_46080˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_conv_3_layer_call_fn_46089˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_conv_3_layer_call_and_return_conditional_losses_46099˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
5__inference_batch_normalization_2_layer_call_fn_46112
5__inference_batch_normalization_2_layer_call_fn_46125
5__inference_batch_normalization_2_layer_call_fn_46138
5__inference_batch_normalization_2_layer_call_fn_46151´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
2˙
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46169
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46187
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46205
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46223´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
,__inference_activation_2_layer_call_fn_46228˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
G__inference_activation_2_layer_call_and_return_conditional_losses_46233˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_conv_4_layer_call_fn_46242˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_conv_4_layer_call_and_return_conditional_losses_46252˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
5__inference_batch_normalization_3_layer_call_fn_46265
5__inference_batch_normalization_3_layer_call_fn_46278
5__inference_batch_normalization_3_layer_call_fn_46291
5__inference_batch_normalization_3_layer_call_fn_46304´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
2˙
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46322
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46340
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46358
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46376´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
,__inference_activation_3_layer_call_fn_46381˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
G__inference_activation_3_layer_call_and_return_conditional_losses_46386˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_conv_5_layer_call_fn_46395˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_conv_5_layer_call_and_return_conditional_losses_46405˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
5__inference_batch_normalization_4_layer_call_fn_46418
5__inference_batch_normalization_4_layer_call_fn_46431
5__inference_batch_normalization_4_layer_call_fn_46444
5__inference_batch_normalization_4_layer_call_fn_46457´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
2˙
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46475
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46493
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46511
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46529´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
,__inference_activation_4_layer_call_fn_46534˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
G__inference_activation_4_layer_call_and_return_conditional_losses_46539˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
(__inference_max_pool_layer_call_fn_43748ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ť2¨
C__inference_max_pool_layer_call_and_return_conditional_losses_43742ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
,__inference_average_pool_layer_call_fn_43761ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ż2Ź
G__inference_average_pool_layer_call_and_return_conditional_losses_43755ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ń2Î
'__inference_dense_1_layer_call_fn_46548˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
B__inference_dense_1_layer_call_and_return_conditional_losses_46558˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ
5__inference_batch_normalization_5_layer_call_fn_46571
5__inference_batch_normalization_5_layer_call_fn_46584´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ţ2Ű
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46604
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46638´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
,__inference_activation_5_layer_call_fn_46643˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
G__inference_activation_5_layer_call_and_return_conditional_losses_46648˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
)__inference_dropout_1_layer_call_fn_46653
)__inference_dropout_1_layer_call_fn_46658´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ć2Ă
D__inference_dropout_1_layer_call_and_return_conditional_losses_46663
D__inference_dropout_1_layer_call_and_return_conditional_losses_46675´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ô2Ń
*__inference_classifier_layer_call_fn_46684˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ď2ě
E__inference_classifier_layer_call_and_return_conditional_losses_46695˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ËBČ
#__inference_signature_wrapper_45305input_im"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 Ë
 __inference__wrapped_model_43106Ś.%&'(1289:;DEKLMNWX^_`ajkqrst:˘7
0˘-
+(
input_im˙˙˙˙˙˙˙˙˙ 
Ş "8Ş5
3

classifier%"

classifier˙˙˙˙˙˙˙˙˙Ä>ľ
G__inference_activation_1_layer_call_and_return_conditional_losses_46080j8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 
,__inference_activation_1_layer_call_fn_46075]8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş "!˙˙˙˙˙˙˙˙˙<ľ
G__inference_activation_2_layer_call_and_return_conditional_losses_46233j8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 
,__inference_activation_2_layer_call_fn_46228]8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş "!˙˙˙˙˙˙˙˙˙<ľ
G__inference_activation_3_layer_call_and_return_conditional_losses_46386j8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 
,__inference_activation_3_layer_call_fn_46381]8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş "!˙˙˙˙˙˙˙˙˙<ľ
G__inference_activation_4_layer_call_and_return_conditional_losses_46539j8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 
,__inference_activation_4_layer_call_fn_46534]8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş "!˙˙˙˙˙˙˙˙˙<Ľ
G__inference_activation_5_layer_call_and_return_conditional_losses_46648Z0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 }
,__inference_activation_5_layer_call_fn_46643M0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ą
E__inference_activation_layer_call_and_return_conditional_losses_45927h7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙<@
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙<@
 
*__inference_activation_layer_call_fn_45922[7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙<@
Ş " ˙˙˙˙˙˙˙˙˙<@Đ
G__inference_average_pool_layer_call_and_return_conditional_losses_43755R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 §
,__inference_average_pool_layer_call_fn_43761wR˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙í
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4601689:;N˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 í
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4603489:;N˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Č
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46052t89:;<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 Č
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_46070t89:;<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 Ĺ
5__inference_batch_normalization_1_layer_call_fn_4595989:;N˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ĺ
5__inference_batch_normalization_1_layer_call_fn_4597289:;N˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
5__inference_batch_normalization_1_layer_call_fn_45985g89:;<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p 
Ş "!˙˙˙˙˙˙˙˙˙< 
5__inference_batch_normalization_1_layer_call_fn_45998g89:;<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p
Ş "!˙˙˙˙˙˙˙˙˙<í
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46169KLMNN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 í
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46187KLMNN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Č
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46205tKLMN<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 Č
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_46223tKLMN<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 Ĺ
5__inference_batch_normalization_2_layer_call_fn_46112KLMNN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ĺ
5__inference_batch_normalization_2_layer_call_fn_46125KLMNN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
5__inference_batch_normalization_2_layer_call_fn_46138gKLMN<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p 
Ş "!˙˙˙˙˙˙˙˙˙< 
5__inference_batch_normalization_2_layer_call_fn_46151gKLMN<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p
Ş "!˙˙˙˙˙˙˙˙˙<í
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46322^_`aN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 í
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46340^_`aN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Č
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46358t^_`a<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 Č
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_46376t^_`a<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 Ĺ
5__inference_batch_normalization_3_layer_call_fn_46265^_`aN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ĺ
5__inference_batch_normalization_3_layer_call_fn_46278^_`aN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
5__inference_batch_normalization_3_layer_call_fn_46291g^_`a<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p 
Ş "!˙˙˙˙˙˙˙˙˙< 
5__inference_batch_normalization_3_layer_call_fn_46304g^_`a<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p
Ş "!˙˙˙˙˙˙˙˙˙<í
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46475qrstN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 í
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46493qrstN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Č
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46511tqrst<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 Č
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46529tqrst<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 Ĺ
5__inference_batch_normalization_4_layer_call_fn_46418qrstN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ĺ
5__inference_batch_normalization_4_layer_call_fn_46431qrstN˘K
D˘A
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
5__inference_batch_normalization_4_layer_call_fn_46444gqrst<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p 
Ş "!˙˙˙˙˙˙˙˙˙< 
5__inference_batch_normalization_4_layer_call_fn_46457gqrst<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙<
p
Ş "!˙˙˙˙˙˙˙˙˙<ź
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46604h4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ź
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46638h4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
5__inference_batch_normalization_5_layer_call_fn_46571[4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙
5__inference_batch_normalization_5_layer_call_fn_46584[4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙é
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45863%&'(M˘J
C˘@
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
p 
Ş "?˘<
52
0+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 é
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45881%&'(M˘J
C˘@
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
p
Ş "?˘<
52
0+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 Ä
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45899r%&'(;˘8
1˘.
(%
inputs˙˙˙˙˙˙˙˙˙<@
p 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙<@
 Ä
N__inference_batch_normalization_layer_call_and_return_conditional_losses_45917r%&'(;˘8
1˘.
(%
inputs˙˙˙˙˙˙˙˙˙<@
p
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙<@
 Á
3__inference_batch_normalization_layer_call_fn_45806%&'(M˘J
C˘@
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
p 
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@Á
3__inference_batch_normalization_layer_call_fn_45819%&'(M˘J
C˘@
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
p
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
3__inference_batch_normalization_layer_call_fn_45832e%&'(;˘8
1˘.
(%
inputs˙˙˙˙˙˙˙˙˙<@
p 
Ş " ˙˙˙˙˙˙˙˙˙<@
3__inference_batch_normalization_layer_call_fn_45845e%&'(;˘8
1˘.
(%
inputs˙˙˙˙˙˙˙˙˙<@
p
Ş " ˙˙˙˙˙˙˙˙˙<@Š
E__inference_classifier_layer_call_and_return_conditional_losses_46695`0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙Ä>
 
*__inference_classifier_layer_call_fn_46684S0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ä>ß
>__inference_cnn_layer_call_and_return_conditional_losses_45114.%&'(1289:;DEKLMNWX^_`ajkqrstB˘?
8˘5
+(
input_im˙˙˙˙˙˙˙˙˙ 
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙Ä>
 ß
>__inference_cnn_layer_call_and_return_conditional_losses_45216.%&'(1289:;DEKLMNWX^_`ajkqrstB˘?
8˘5
+(
input_im˙˙˙˙˙˙˙˙˙ 
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙Ä>
 Ý
>__inference_cnn_layer_call_and_return_conditional_losses_45610.%&'(1289:;DEKLMNWX^_`ajkqrst@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙ 
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙Ä>
 Ý
>__inference_cnn_layer_call_and_return_conditional_losses_45774.%&'(1289:;DEKLMNWX^_`ajkqrst@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙ 
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙Ä>
 ˇ
#__inference_cnn_layer_call_fn_44318.%&'(1289:;DEKLMNWX^_`ajkqrstB˘?
8˘5
+(
input_im˙˙˙˙˙˙˙˙˙ 
p 

 
Ş "˙˙˙˙˙˙˙˙˙Ä>ˇ
#__inference_cnn_layer_call_fn_45012.%&'(1289:;DEKLMNWX^_`ajkqrstB˘?
8˘5
+(
input_im˙˙˙˙˙˙˙˙˙ 
p

 
Ş "˙˙˙˙˙˙˙˙˙Ä>ľ
#__inference_cnn_layer_call_fn_45386.%&'(1289:;DEKLMNWX^_`ajkqrst@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙ 
p 

 
Ş "˙˙˙˙˙˙˙˙˙Ä>ľ
#__inference_cnn_layer_call_fn_45467.%&'(1289:;DEKLMNWX^_`ajkqrst@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙ 
p

 
Ş "˙˙˙˙˙˙˙˙˙Ä>˛
A__inference_conv_1_layer_call_and_return_conditional_losses_45793m8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙ 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙<@
 
&__inference_conv_1_layer_call_fn_45783`8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙ 
Ş " ˙˙˙˙˙˙˙˙˙<@˛
A__inference_conv_2_layer_call_and_return_conditional_losses_45946m127˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙<@
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 
&__inference_conv_2_layer_call_fn_45936`127˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙<@
Ş "!˙˙˙˙˙˙˙˙˙<ł
A__inference_conv_3_layer_call_and_return_conditional_losses_46099nDE8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 
&__inference_conv_3_layer_call_fn_46089aDE8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş "!˙˙˙˙˙˙˙˙˙<ł
A__inference_conv_4_layer_call_and_return_conditional_losses_46252nWX8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 
&__inference_conv_4_layer_call_fn_46242aWX8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş "!˙˙˙˙˙˙˙˙˙<ł
A__inference_conv_5_layer_call_and_return_conditional_losses_46405njk8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙<
 
&__inference_conv_5_layer_call_fn_46395ajk8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙<
Ş "!˙˙˙˙˙˙˙˙˙<Ś
B__inference_dense_1_layer_call_and_return_conditional_losses_46558`0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
'__inference_dense_1_layer_call_fn_46548S0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ś
D__inference_dropout_1_layer_call_and_return_conditional_losses_46663^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ś
D__inference_dropout_1_layer_call_and_return_conditional_losses_46675^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
)__inference_dropout_1_layer_call_fn_46653Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙~
)__inference_dropout_1_layer_call_fn_46658Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙ć
C__inference_max_pool_layer_call_and_return_conditional_losses_43742R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ž
(__inference_max_pool_layer_call_fn_43748R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ú
#__inference_signature_wrapper_45305˛.%&'(1289:;DEKLMNWX^_`ajkqrstF˘C
˘ 
<Ş9
7
input_im+(
input_im˙˙˙˙˙˙˙˙˙ "8Ş5
3

classifier%"

classifier˙˙˙˙˙˙˙˙˙Ä>