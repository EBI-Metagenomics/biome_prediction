#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH/model.p.gz
curl -o $SCRIPTPATH/model.p.gz ftp://ftp.ebi.ac.uk/pub/databases/metagenomics/biome_prediction_models/model.p.gz
