#!/bin/sh
set -eu

# Create app database, schemas, and user if they don't exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL

  SELECT format('CREATE DATABASE %I OWNER %I', '${POSTGRES_DB_NAME}', '${POSTGRES_USER}')
  WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = '${POSTGRES_DB_NAME}'
  )
  \gexec

EOSQL
