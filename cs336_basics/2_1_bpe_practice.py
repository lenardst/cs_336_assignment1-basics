# Problem 1

chr(0)
# '\x00'
char_str = chr(0)
print(char_str)
# Returns the null character that indicates an empty value, which can be used differently in different cases. 

print(char_str.__repr__())
# The printed representation is simply empty. The string representation is \x00 which is o in hexadecimal notation.

"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")
# It is printed in its string representation (hex format) when executed in the python terminal simply not printed when passed to print.

# Problem 2
test_string = "hello! Tschüss"
utf8_encoded = test_string.encode('utf-8')
print(utf8_encoded)
utf16_encoded = test_string.encode('utf-16')
utf32_encoded = test_string.encode('utf-32')
print(list(utf8_encoded))
print(list(utf16_encoded))
print(list(utf32_encoded))

# UTF-8 encoding required only half or a fourth of the bytes of the richer encodings. This might make training faster. Also, UTF-8 encoding does support even more than 264 characters because some encodings use more than 1 byte. It is variable in the number of bytes it uses. However, frequent characters can be represented as on character, which saves storage and might speed up training.

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

utf8_decoded = decode_utf8_bytes_to_str_wrong(utf8_encoded)
print(utf8_decoded)

# For the string "hello! Tschüss", the function does not return the correct original string because the letter ü is encoded into two bytes. The function, however, assumes that each character corresponds to one byte.

# 11000000 11000000
# 110 signals a two byte sequence. But it cannot be signaled twice. The continuation byte needs to start with 10. This sequence is therefore invalid. 