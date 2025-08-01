#ifndef CSV_IMPORTER_H
#define CSV_IMPORTER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVImporter
{
private:
    std::vector<std::vector<double>> table;

    // This function splits a string by a delimiter and returns a vector of numeric values
    std::vector<double> tokenize(const std::string& s, char delimiter);

public:
    CSVImporter(const std::string &filename, char delimiter = ',');
    const std::vector<double>& getRow(size_t index) const;
    const std::vector<std::vector<double>>& getTable() const;
    size_t numRows() const;
    size_t numColumns(size_t rowIndex) const;
};

#endif // CSV_IMPORTER_H
