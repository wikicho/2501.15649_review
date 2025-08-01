#include "csv_importer.h"

CSVImporter::CSVImporter(const std::string &filename, char delimiter)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip lines starting with #
        if (!line.empty() && line[0] == '#') {
            continue;
        }
        table.push_back(tokenize(line, delimiter));
    }

    file.close();
}

std::vector<double> CSVImporter::tokenize(const std::string &s, char delimiter)
{
    std::vector<double> tokens;
    double tokenValue;
    std::istringstream tokenStream(s);
    std::string token;

    while (std::getline(tokenStream, token, delimiter))
    {
        try {
            tokenValue = std::stod(token);
            tokens.push_back(tokenValue);
        } catch (const std::out_of_range& e) {
            std::cerr << "Error converting string to double: " << token << '\n';
        }
    }

    return tokens;
}

const std::vector<double> &CSVImporter::getRow(size_t index) const
{
    return table.at(index);
}

const std::vector<std::vector<double>> &CSVImporter::getTable() const
{
    return table;
}

size_t CSVImporter::numRows() const
{
    return table.size();
}

size_t CSVImporter::numColumns(size_t rowIndex) const
{
    return table.at(rowIndex).size();
}