for predicting compute reduction to achieve a fixed capability:
filter to models within a narrow ECI bucket (3 points? you’d sweep over this to see sensitivity to the bucket size i guess)
filter to models which are “SOTA in compute efficiency at release” within that bucket (i.e. they use less compute than any prior model in that bucket)
do a linear fit of compute over time within that bucket. The slope is the compute reduction in OOMs/year to achieve that capability.
repeat for other buckets
take the mean or median of the slopes???
for predicting capability gains for a fixed level of compute:
filter to models within a narrow compute bucket (e.g. 0.3 OOMs)
filter models within that compute bucket which were SOTA at release date
do a linear fit of ECI over time within that bucket. The slope is the number of capability units/year gained at that level of compute.
repeat for other buckets
take the mean or median slope???
