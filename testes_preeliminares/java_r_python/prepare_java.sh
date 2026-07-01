#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "==> Checking prerequisites..."

command -v java >/dev/null || {
    echo "Error: Java is not installed."
    exit 1
}

command -v mvn >/dev/null || {
    echo "Error: Maven is not installed."
    exit 1
}

echo "==> Java version"
java -version

echo
echo "==> Maven version"
mvn -version

echo
echo "==> Cleaning previous build..."
mvn clean

echo
echo "==> Downloading project dependencies..."
mvn dependency:resolve

echo
echo "==> Copying runtime dependencies..."
mvn dependency:copy-dependencies \
    -DincludeScope=runtime

echo
echo "==> Compiling and packaging..."
mvn package

echo
echo "==> Verifying build..."

MAIN_CLASS="target/classes/benchmark/MOABenchmark.class"

if [[ ! -f "$MAIN_CLASS" ]]; then
    echo "ERROR: Main class not found:"
    echo "  $MAIN_CLASS"
    exit 1
fi

echo "Main class found."

echo
echo "Runtime JARs:"
find target/dependency -name '*.jar' | sort

echo
echo "Build completed successfully."
