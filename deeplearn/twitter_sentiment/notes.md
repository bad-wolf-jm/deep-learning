
|Number of bytes|	Bits for code point	| First code point | Last code point | Byte 1 |	Byte 2 | Byte 3 | Byte 4 |
|---------------|---------------------|------------------|-----------------|--------|--------|--------|--------|
|1	| 7  |U+0000 |U+007F	|0xxxxxxx	|		
|2	| 11 |U+0080 |U+07FF	|110xxxxx	|10xxxxxx
|3	| 16 |U+0800 |U+FFFF	|1110xxxx	|10xxxxxx	|10xxxxxx
|4	| 21 |U+10000	|U+10FFFF	|11110xxx	|10xxxxxx	|10xxxxxx	|10xxxxxx
