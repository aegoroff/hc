cls
win_flex.exe --wincompat --outfile="linq2hash.flex.c" linq2hash.lex
win_bison --output="linq2hash.tab.c" -dy linq2hash.y
