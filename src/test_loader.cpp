# include "loader.hpp"
# include <iostream>

using namespace std;

int main() {
    load_weights_from_json("../data/inceptionV3.json", true);
    return 0;
}