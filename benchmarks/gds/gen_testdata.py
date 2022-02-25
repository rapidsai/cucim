import secrets


def gen_data(data_folder: str):
    for file_size in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        file_name = f"{data_folder}/data_{file_size}.blob"
        print(f"Generating {file_name}...")
        with open(file_name, 'wb') as fp:
            for i in range(int(file_size/0.1)):
                fp.write(secrets.token_bytes(int(2**30 * 0.1)))


if __name__ == "__main__":
    gen_data(".")
