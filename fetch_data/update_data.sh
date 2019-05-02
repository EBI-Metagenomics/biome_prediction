#!/usr/bin/env bash
HOST="mysql-pgvm-012.ebi.ac.uk"
USER="emguser"
PORT="4501"
PASSWORD="emguser"

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"


mysql -u $USER --password=$PASSWORD -h $HOST -P $PORT < $SCRIPTPATH/fetch.sql | sed 's/NULL/ /g' > $SCRIPTPATH/raw_data.tsv
mv $SCRIPTPATH/raw_data.tsv $SCRIPTPATH/../data/raw_data.tsv
gzip $SCRIPTPATH/../data/raw_data.tsv