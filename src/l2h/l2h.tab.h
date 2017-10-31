/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_L2H_TAB_H_INCLUDED
# define YY_YY_L2H_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif
/* "%code requires" blocks.  */
#line 14 "l2h.y" /* yacc.c:1909  */

	#include "lib.h"
	#include "frontend.h"

#line 49 "l2h.tab.h" /* yacc.c:1909  */

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    COMMENT = 258,
    SEMICOLON = 259,
    FROM = 260,
    TYPE = 261,
    LET = 262,
    ASSIGN = 263,
    WHERE = 264,
    ON = 265,
    EQUALS = 266,
    JOIN = 267,
    ORDERBY = 268,
    COMMA = 269,
    ASCENDING = 270,
    DESCENDING = 271,
    SELECT = 272,
    GRP = 273,
    BY = 274,
    OPEN_PAREN = 275,
    CLOSE_PAREN = 276,
    OPEN_BRACE = 277,
    CLOSE_BRACE = 278,
    IDENTIFIER = 279,
    INTEGER = 280,
    STRING = 281,
    DOT = 282,
    INVALID_STRING = 283,
    LOWER_THAN_INTO = 284,
    INTO = 285,
    OR = 286,
    AND = 287,
    WITHIN = 288,
    NOT = 289,
    REL_OP = 290
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE YYSTYPE;
union YYSTYPE
{
#line 21 "l2h.y" /* yacc.c:1909  */

	cond_op_t relational_op;
	ordering_t ordering;
	long long number;
	char* string;
	type_info_t* type;
	fend_node_t* node;

#line 106 "l2h.tab.h" /* yacc.c:1909  */
};
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


extern YYSTYPE yylval;
extern YYLTYPE yylloc;
int yyparse (void);

#endif /* !YY_YY_L2H_TAB_H_INCLUDED  */
