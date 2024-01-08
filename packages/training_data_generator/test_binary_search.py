from packages.training_data_generator.generator import BinarySearch

if __name__ == "__main__":
    l, r = BinarySearch.binary_search_range(
        [1, 2, 5, 7, 8, 9, 13, 14, 16],
        3, 12
    )
    print(l, r) # 2 6

    l, r = BinarySearch.binary_search_range(
        [1, 2, 5, 7, 8, 9, 13, 14, 16],
        1, 9
    )
    print(l, r) # 0 6

    l, r = BinarySearch.binary_search_range(
        [1, 2, 5, 7, 8, 9, 13, 14, 16],
        2, 2
    )
    print(l, r) # 1 2