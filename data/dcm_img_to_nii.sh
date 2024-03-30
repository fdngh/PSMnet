#!bin/bash
# Convert dicom-like images to nii files in 3D
# This is the first step for image pre-processing

# Feed path to the downloaded data here
DATAPATH=~/.tt/aa/Self-supervised-Fewshot-Medical-Image-Segmentation-master/data/CHAOST2/MR
# please put chaos dataset training fold here which contains ground truth

# Feed path to the output folder here
OUTPATH=~/.tt/aa/Self-supervised-Fewshot-Medical-Image-Segmentation-master/data/CHAOST2/niis

if [ ! -d  $OUTPATH/T2SPIR ]
then
    mkdir $OUTPATH/T2SPIR
fi

for sid in $(ls "$DATAPATH")
do
	dcm2nii -o "$DATAPATH/$sid/T2SPIR" "$DATAPATH/$sid/T2SPIR/DICOM_anon";
	find "$DATAPATH/$sid/T2SPIR" -name "*.nii.gz" -exec mv {} "$OUTPATH/T2SPIR/image_$sid.nii.gz" \;
done;


