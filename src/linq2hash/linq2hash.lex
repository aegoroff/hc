/* recognize tokens for the calculator and print them out */

%option noyywrap 
%{
    #include <stdlib.h>
	#include "frontend.h"
    #include "linq2hash.tab.h"
%}

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

STR_ESCAPE_SEQ ("\\".)

STRING ("'"({STR_ESCAPE_SEQ}|[^\\\r\n'])*"'"|"\""({STR_ESCAPE_SEQ}|[^\\\r\n"])*"\"")

WS [ \t\v\f]
ENDL [\r\n]

%%

{FROM} { return FROM; }
{GRP} { return GRP; }
{BY} { return BY; }
{OR} { return OR; }
{AND} { return AND; }
{NOT} { return NOT; }
{ORDERBY} { return ORDERBY; }
{ASCENDING} { yylval.Ordering = ordering_asc; return ASCENDING; }
{DESCENDING} { yylval.Ordering = ordering_desc; return DESCENDING; }

{FILE} { yylval.Type = fend_on_simple_type_def(type_def_file); return TYPE; }
{STRING_TYPE} { yylval.Type = fend_on_simple_type_def(type_def_string); return TYPE; }
{DIR} { yylval.Type = fend_on_simple_type_def(type_def_dir); return TYPE; }

{SELECT} { return SELECT; }
{INTO} { return INTO; }
{JOIN} { return JOIN; }
{WITHIN} { return WITHIN; }
{ON} { return ON; }
{ASSIGN} { return ASSIGN; }
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

{EQUAL} { yylval.RelOp = cond_op_eq; return REL_OP; }
{NOTEQUAL} { yylval.RelOp = cond_op_not_eq; return REL_OP; }

{GT} { yylval.RelOp = cond_op_ge; return REL_OP; }
{GE} { yylval.RelOp = cond_op_ge_eq; return REL_OP; }
{LT} { yylval.RelOp = cond_op_le; return REL_OP; }
{LE} { yylval.RelOp = cond_op_le_eq; return REL_OP; }
{MATCH} { yylval.RelOp = cond_op_match; return REL_OP; }
{NOTMATCH} { yylval.RelOp = cond_op_not_match; return REL_OP; }
{WS} { }
{ENDL} { yylineno++; }

{IDENTIFIER} { yylval.String = fend_query_strdup(yytext); return IDENTIFIER; }
{DIGIT}+ { yylval.Number = fend_to_number(yytext); return INTEGER; }
{STRING} { yylval.String = fend_query_strdup(yytext); return STRING; }

%%
