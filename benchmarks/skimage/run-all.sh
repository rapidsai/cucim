for file in ./cu*py
do
  echo $file
  time python "$file" 
done
