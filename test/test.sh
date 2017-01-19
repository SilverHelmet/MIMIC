max_length=256
str=""
length=1
while  [ $length -le $max_length ]
do
    echo "length = $length"
    length=$(($length * 2))
done