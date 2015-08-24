/* recognize tokens for the calculator and print them out */

%option noyywrap 
%{
    #include <stdlib.h>
	#include "frontend.h"
    #include "linq2hash.tab.h"
%}

MD2 "md2"
MD4 "md4"
MD5 "md5"
SHA1 "sha1"
SHA224 "sha224"
SHA384 "sha384"
SHA512 "sha512"
CRC32 "crc32"
WHIRLPOOL "whirlpool"
TIGER "tiger"
TIGER2 "tiger2"
RIPEMD128 "ripemd128"
RIPEMD160 "ripemd160"
RIPEMD256 "ripemd256"
RIPEMD320 "ripemd320"
GOST "gost"
SNEFRU128 "snefru128"
SNEFRU256 "snefru256"
TTH "tth"
HAVAL_128_3 "haval-128-3"
HAVAL_128_4 "haval-128-4"
HAVAL_128_5 "haval-128-5"
HAVAL_160_3 "haval-160-3"
HAVAL_160_4 "haval-160-4"
HAVAL_160_5 "haval-160-5"
HAVAL_192_3 "haval-192-3"
HAVAL_192_4 "haval-192-4"
HAVAL_192_5 "haval-192-5"
HAVAL_224_3 "haval-224-3"
HAVAL_224_4 "haval-224-4"
HAVAL_224_5 "haval-224-5"
HAVAL_256_3 "haval-256-3"
HAVAL_256_4 "haval-256-4"
HAVAL_256_5 "haval-256-5"
EDONR256 "edonr256"
EDONR512 "edonr512"
NTLM "ntlm"
SHA_3_224 "sha-3-224"
SHA_3_256 "sha-3-256"
SHA_3_384 "sha-3-384"
SHA_3_512 "sha-3-512"
SHA_3K_224 "sha-3k-224"
SHA_3K_256 "sha-3k-256"
SHA_3K_384 "sha-3k-384"
SHA_3K_512 "sha-3k-512"
FILE file
STRING_TYPE string
DIR dir
HASH ({MD2}|{MD4}|{MD5}|{SHA1}|{SHA224}|{SHA384}|{SHA512}|{CRC32}|{WHIRLPOOL}|{TIGER}|{TIGER2}|{RIPEMD128}|{RIPEMD160}|{RIPEMD256}|{RIPEMD320}|{GOST}|{SNEFRU128}|{SNEFRU256}|{TTH}|{HAVAL_128_3}|{HAVAL_128_4}|{HAVAL_128_5}|{HAVAL_160_3}|{HAVAL_160_4}|{HAVAL_160_5}|{HAVAL_192_3}|{HAVAL_192_4}|{HAVAL_192_5}|{HAVAL_224_3}|{HAVAL_224_4}|{HAVAL_224_5}|{HAVAL_256_3}|{HAVAL_256_4}|{HAVAL_256_5}|{EDONR256}|{EDONR512}|{NTLM}|{SHA_3_224}|{SHA_3_256}|{SHA_3_384}|{SHA_3_512}|{SHA_3K_224}|{SHA_3K_256}|{SHA_3K_384}|{SHA_3K_512})


FROM "from"
WITHIN "in"
ON "on"
LET "let"
WHERE "where"
SELECT "select"
INTO "into"
GRP "group"
JOIN "join"
BY "by"
OR "or"
AND "and"
NOT "not"
EQUALS "equals"
DOT "."
COMMA ","
ORDERBY "orderby"
ASCENDING "ascending"
DESCENDING "descending"
SEMICOLON ";"
OPEN_PAREN "("
OPEN_BRACE "{"
CLOSE_PAREN ")"
CLOSE_BRACE "}"
UPPER_LETTER [A-Z]
LOWER_LETTER [a-z]
DIGIT [0-9]

EQ "="
EQUAL "=="
NOTEQUAL "!="
GT ">"
GE ">="
LT "<"
LE "<="
MATCH "~"
NOTMATCH "!~"

ID_START  (_|[a-zA-Z])
ID_PART   ({ID_START}|{DIGIT})
IDENTIFIER ({ID_START}{ID_PART}*)

STR_ESCAPE_SEQ ("\\".)

STRING ("'"({STR_ESCAPE_SEQ}|[^\\\r\n'])*"'"|"\""({STR_ESCAPE_SEQ}|[^\\\r\n"])*"\"")

WS [ \t\v\f\r\n]

%%

{FROM} { return FROM; }
{GRP} { return GRP; }
{BY} { return BY; }
{OR} { return OR; }
{AND} { return AND; }
{NOT} { return NOT; }
{ORDERBY} { return ORDERBY; }
{ASCENDING} { yylval.Ordering = OrderingAsc; return ASCENDING; }
{DESCENDING} { yylval.Ordering = OrderingDesc; return DESCENDING; }

{FILE} { yylval.Info = OnSimpleTypeDef(TypeDefFile); return TYPE; }
{STRING_TYPE} { yylval.Info = OnSimpleTypeDef(TypeDefString); return TYPE; }
{DIR} { yylval.Info = OnSimpleTypeDef(TypeDefDir); return TYPE; }
{HASH} { yylval.Info = OnComplexTypeDef(TypeDefHash, yytext); return TYPE; }

{SELECT} { return SELECT; }
{INTO} { return INTO; }
{JOIN} { return JOIN; }
{WITHIN} { return WITHIN; }
{ON} { return ON; }
{EQ} { return EQ; }
{EQUALS} { return EQUALS; }
{LET} { return LET; }
{DOT} { return DOT; }
{WHERE} { return WHERE; }
{SEMICOLON} { return SEMICOLON; }
{DOT} { return DOT; }
{COMMA} { return COMMA; }
{OPEN_PAREN} { return OPEN_PAREN; }
{CLOSE_PAREN} { return CLOSE_PAREN; }
{OPEN_BRACE} { return OPEN_BRACE; }
{CLOSE_BRACE} { return CLOSE_BRACE; }

{EQUAL} { yylval.RelOp = CondOpEq; return REL_OP; }
{NOTEQUAL} { yylval.RelOp = CondOpNotEq; return REL_OP; }

{GT} { yylval.RelOp = CondOpGe; return REL_OP; }
{GE} { yylval.RelOp = CondOpGeEq; return REL_OP; }
{LT} { yylval.RelOp = CondOpLe; return REL_OP; }
{LE} { yylval.RelOp = CondOpLeEq; return REL_OP; }
{MATCH} { yylval.RelOp = CondOpMatch; return REL_OP; }
{NOTMATCH} { yylval.RelOp = CondOpNotMatch; return REL_OP; }
{WS} { }

{IDENTIFIER} { yylval.String = QueryStrdup(yytext); return IDENTIFIER; }
{DIGIT}+ { yylval.Number = ToNumber(yytext); return INTEGER; }
{STRING} { yylval.String = QueryStrdup(yytext); return STRING; }

%%
