#!/bin/bash

if [ "$1" = "train" ]; then
  if [ "$2" = "all" ]; then
      python run.py train --num_topics=40 --all
  elif [ "$2" = "" ]; then
      python run.py train --num_topics=18 --days=1
  fi
elif [ "$1" = "predict" ]; then
  python run.py predict --num_topics=18 --days=1
elif [ "$1" = "update" ]; then
  if [ "$2" = "all" ]; then
      python run.py update --num_topics=40 --all --days=1
  elif [ "$2" = "" ]; then
      python run.py update --num_topics=18 --days=1
  fi
elif [ "$1" = "delete" ]; then
  rm ./tmp/new/new_topic*
elif [ "$1" = "dput" ]; then
  if [ "$2" = "all" ]; then
      for (( i = 89; i > 0; i-- )); do
      rm ./tmp/new/new_topic*
      python run.py predict --num_topics=18 --days="$i"
      python run.py update --num_topics=40 --all --days="$i"
      python run.py train --num_topics=40 --all --days="$i"
    done
  elif [ "$2" = "one" ]; then
      rm ./tmp/new/new_topic*
      python run.py predict --num_topics=18 --days=1
      python run.py update --num_topics=18 --days=1
      python run.py train --num_topics=18 --days=1
  elif [ "$2" = "" ]; then
      for (( i = 89; i > 0; i-- )); do
      rm ./tmp/new/new_topic*
      python run.py predict --num_topics=18 --days="$i"
      python run.py update --num_topics=18 --days="$i"
      python run.py train --num_topics=18 --days="$i"
    done
  fi
else
	echo "Invalid Option Selected"
fi


