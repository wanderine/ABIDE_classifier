#!/bin/bash

# Get control subjects
#awk -F, '$8 == 2' Phenotypic_V1_0b_preprocessed1.csv > controls.txt

# Get ASD subjects
#awk -F, '$8 == 1' Phenotypic_V1_0b_preprocessed1.csv > asd.txt

#cat controls.txt | cut --delimiter=, -f7 > control_subjects.txt

#cat asd.txt | cut --delimiter=, -f7 > asd_subjects.txt

MaxThreads=10

# First download ASD, DX 2

# First download controls, DX 1

# Loop over control subjects
temp=`cat control_subjects.txt`
Subjects=()
subjectstring=${temp[$((0))]}
Subjects+=($subjectstring)

threads=0
for SubjectNumber in {0..572}; do

    Subject=${Subjects[$((${SubjectNumber}))]}
    echo "Now processing $SubjectNumber $Subject"
    wget https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/${Subject}/mri/brain.mgz
    mv brain.mgz CONTROL_${Subject}_brain.mgz
    /flush2/common/freesurfer/bin/mri_convert --in_type mgz --out_type nii CONTROL_${Subject}_brain.mgz CONTROL_${Subject}_brain.nii.gz

    ((threads++))
    if [ "$threads" -eq "$MaxThreads" ]; then
        wait
	threads=0
    fi

done

# Loop over ASD subjects

temp=`cat asd_subjects.txt`
Subjects=()
subjectstring=${temp[$((0))]}
Subjects+=($subjectstring)

threads=0
for SubjectNumber in {0..538}; do

    Subject=${Subjects[$((${SubjectNumber}))]}
    echo "Now processing $Subject"
    wget https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/${Subject}/mri/brain.mgz
    mv brain.mgz ASD_${Subject}_brain.mgz
    /flush2/common/freesurfer/bin/mri_convert --in_type mgz --out_type nii ASD_${Subject}_brain.mgz ASD_${Subject}_brain.nii.gz

    ((threads++))
    if [ "$threads" -eq "$MaxThreads" ]; then
        wait
	threads=0
    fi

done

mv ASD_* ASDS
mv CONTROL_* CONTROLS



