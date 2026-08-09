/* recognize tokens for the calculator and print them out */

%{
    #include <stdlib.h>
	#include "frontend.h"
    #include "l2h.tab.h"

	/* handle locations */
	int yycolumn = 1;

/* Bison YYLTYPE columns are half-open [first, last). After a newline,
 * last_column must be 1 so the next line's first character is column 1
 * (last_column = 0 shifts every column on the following line by -1). */
#define YY_USER_ACTION \
    yylloc.first_line = yylloc.last_line; \
    yylloc.first_column = yylloc.last_column; \
    for(int i = 0; yytext[i] != '\0'; i++) { \
        if(yytext[i] == '\n') { \
            yylloc.last_line++; \
            yylloc.last_column = 1; \
        } \
        else { \
            yylloc.last_column++; \
        } \
    }
%}

%option noyywrap 
%option yylineno

%s DEFINITION

FILE file
STRING_TYPE string
DIR dir

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
OR "||"
AND "&&"
NOT "!"
BOOL_TRUE "true"
BOOL_FALSE "false"
UPPER_LETTER [A-Z]
LOWER_LETTER [a-z]
DIGIT [0-9]

ASSIGN "="
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

COMMENT ^#[^\r\n]*

/* Ordinary strings are raw (paths like 'c:\Windows'). Escapes only in b'…' / b"…'. */
STRING_CHAR_SQ ([^\'\\\r\n]|\\.)
STRING_CHAR_DQ ([^\"\\\r\n]|\\.)
BYTE_STRING ("b'"{STRING_CHAR_SQ}*"'"|"b\""{STRING_CHAR_DQ}*"\"")
STRING ("'"([^\r\n'])*"'"|"\""([^\r\n"])*"\"")

WS [ \t\v\f]
ENDL [\r\n]

%%

{FROM} { BEGIN DEFINITION; return FROM; }
{GRP} { return GRP; }
{BY} { return BY; }
{OR} { return OR; }
{AND} { return AND; }
{NOT} { return NOT; }
{BOOL_TRUE} { return BOOL_TRUE; }
{BOOL_FALSE} { return BOOL_FALSE; }
{ORDERBY} { return ORDERBY; }
{ASCENDING} { yylval.ordering = ordering_asc; return ASCENDING; }
{DESCENDING} { yylval.ordering = ordering_desc; return DESCENDING; }

<DEFINITION>{FILE} { yylval.type = type_def_file;  BEGIN INITIAL; return TYPE; }
<DEFINITION>{STRING_TYPE} { yylval.type = type_def_string;  BEGIN INITIAL; return TYPE; }
<DEFINITION>{DIR} { yylval.type = type_def_dir;  BEGIN INITIAL; return TYPE; }
<DEFINITION>{IDENTIFIER} { yylval.type = type_def_custom; BEGIN INITIAL; return TYPE; }

{SELECT} { return SELECT; }
{INTO} { return INTO; }
{JOIN} { BEGIN DEFINITION; return JOIN; }
{WITHIN} { BEGIN INITIAL; return WITHIN; }
{ON} { return ON; }
{ASSIGN} { return ASSIGN; }
{EQUALS} { return EQUALS; }
{LET} { return LET; }
{DOT} { return DOT; }
{WHERE} { return WHERE; }
{SEMICOLON} { return SEMICOLON; }
{COMMA} { return COMMA; }
{OPEN_PAREN} { return OPEN_PAREN; }
{CLOSE_PAREN} { return CLOSE_PAREN; }
{OPEN_BRACE} { return OPEN_BRACE; }
{CLOSE_BRACE} { return CLOSE_BRACE; }

{EQUAL} { yylval.relational_op = cond_op_eq; return REL_OP; }
{NOTEQUAL} { yylval.relational_op = cond_op_not_eq; return REL_OP; }

{GT} { yylval.relational_op = cond_op_ge; return REL_OP; }
{GE} { yylval.relational_op = cond_op_ge_eq; return REL_OP; }
{LT} { yylval.relational_op = cond_op_le; return REL_OP; }
{LE} { yylval.relational_op = cond_op_le_eq; return REL_OP; }
{MATCH} { yylval.relational_op = cond_op_match; return REL_OP; }
{NOTMATCH} { yylval.relational_op = cond_op_not_match; return REL_OP; }
{WS} { }
{COMMENT} { return COMMENT; }
{ENDL} { yycolumn = 1; }

{IDENTIFIER} { yylval.string = fend_query_strdup(yytext); return IDENTIFIER; }
{DIGIT}+ { yylval.number = fend_to_number(yytext); return INTEGER; }
-{DIGIT}+ { yylval.number = fend_to_number(yytext); return INTEGER; }
{BYTE_STRING} { yylval.string = fend_query_strdup(yytext); return BYTE_STRING; }
{STRING} { yylval.string = fend_query_strdup(yytext); return STRING; }

.  { yylval.string = fend_query_strdup(yytext); return INVALID_STRING; }
%%
