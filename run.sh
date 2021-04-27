#!/bin/bash

perform_mode="train"
all_flag="false"
n_topics=60
days_before=1

func() {
    echo "Usage:"
    echo "run.sh [-t|-u|-p|-r|-w|-o] [-a] [-n n_topics] [-d days]"
}

while getopts "tuprwoan:d:" OPT; do
    case $OPT in
        t) perform_mode="train";;
        u) perform_mode="update";;
        p) perform_mode="predict";;
        r) perform_mode="remove";;
        w) perform_mode="whole";;
        o) perform_mode="one";;
        a) all_flag="true";;
        n) n_topics=$OPTARG;;
        d) days_before=$OPTARG;;
        ?) func;;
    esac
done

if [ $perform_mode = "train" ]; then
    if [ $all_flag = "true" ]; then
      python run.py train --num_topics="$n_topics" --all # all训练用不到days_before
    else
      echo "train model must use 'all' option"
    fi
elif [ $perform_mode = "update" ]; then
    if [ $all_flag = "true" ]; then
      python run.py update --num_topics="$n_topics" --all --days="$days_before"
    else
      echo "update model must use 'all' option"
    fi
elif [ $perform_mode = "predict" ]; then
    if [ "$days_before" = 1 ]; then
        python run.py predict --num_topics=20 --days="$days_before"
    fi
elif [ $perform_mode = "delete" ]; then
    rm ./tmp/new/new_topic*
elif [ $perform_mode = "whole" ]; then
  if [ "$days_before" != 1 ]; then
      if [ $all_flag = "true" ]; then
      for (( i = "$days_before"; i > 0; i-- )); do
        rm ./tmp/new/new_topic*
        python run.py predict --num_topics=20 --days="$i"
        python run.py update --num_topics="$n_topics" --all --days="$i"
        python run.py train --num_topics="$n_topics" --all --days="$i"
      done
    else
      for (( i = "$days_before"; i > 0; i-- )); do
        rm ./tmp/new/new_topic*
        python run.py predict --num_topics=20 --days="$i"
        python run.py update --num_topics=20 --days="$i"
        python run.py train --num_topics=20 --days="$i"
      done
    fi
  fi
elif [ $perform_mode = "one" ]; then
  if [ "$days_before" = 1 ]; then
    rm ./tmp/new/new_topic*
    python run.py predict --num_topics=20 --days="$days_before"
    python run.py update --num_topics=20 --days="$days_before"
    python run.py train --num_topics=20 --days="$days_before"
  fi
else
  echo "Invalid Option Selected"
fi

#if [ "$1" = "train" ]; then
#  if [ "$2" = "all" ]; then
#    if [ "$3" != "" ]; then
#        python run.py train --num_topics="$3" --all
#    else
#      python run.py train --num_topics=60 --all
#    fi
#  elif [ "$2" = "" ]; then
#      python run.py train --num_topics=20 --days=1
#  fi
#elif [ "$1" = "predict" ]; then
#  python run.py predict --num_topics=20 --days=1
#elif [ "$1" = "update" ]; then
#  if [ "$2" = "all" ]; then
#    if [ "$3" != "" ]; then
#        python run.py update --num_topics="$3" --all --days=1
#    else
#      python run.py update --num_topics=60 --all --days=1
#    fi
#  elif [ "$2" = "" ]; then
#      python run.py update --num_topics=20 --days=1
#  fi
#elif [ "$1" = "delete" ]; then
#  rm ./tmp/new/new_topic*
#elif [ "$1" = "dput" ]; then
#  if [ "$2" = "all" ]; then
#    if [ "$3" != "" ]; then
#        for (( i = "$3"; i > 0; i-- )); do
#      rm ./tmp/new/new_topic*
#      python run.py predict --num_topics=20 --days="$i"
#      python run.py update --num_topics=60 --all --days="$i"
#      python run.py train --num_topics=60 --all --days="$i"
#    done
#    fi
#  elif [ "$2" = "one" ]; then
#      rm ./tmp/new/new_topic*
#      python run.py predict --num_topics=20 --days=1
#      python run.py update --num_topics=20 --days=1
#      python run.py train --num_topics=20 --days=1
#  elif [ "$2" = "" ]; then
#    if [ "$3" != "" ]; then
#        for (( i = "$3"; i > 0; i-- )); do
#      rm ./tmp/new/new_topic*
#      python run.py predict --num_topics=20 --days="$i"
#      python run.py update --num_topics=20 --days="$i"
#      python run.py train --num_topics=20 --days="$i"
#    done
#    fi
#  fi
#else
#	echo "Invalid Option Selected"
#fi
