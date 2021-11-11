/* A Bison parser, made by GNU Bison 3.7.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_L2H_TAB_H_INCLUDED
# define YY_YY_L2H_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif
/* "%code requires" blocks.  */
#line 14 "l2h.y"

	#include "lib.h"
	#include "frontend.h"

#line 54 "l2h.tab.h"

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    COMMENT = 258,                 /* COMMENT  */
    SEMICOLON = 259,               /* SEMICOLON  */
    FROM = 260,                    /* FROM  */
    TYPE = 261,                    /* TYPE  */
    LET = 262,                     /* LET  */
    ASSIGN = 263,                  /* ASSIGN  */
    WHERE = 264,                   /* WHERE  */
    ON = 265,                      /* ON  */
    EQUALS = 266,                  /* EQUALS  */
    JOIN = 267,                    /* JOIN  */
    ORDERBY = 268,                 /* ORDERBY  */
    COMMA = 269,                   /* COMMA  */
    ASCENDING = 270,               /* ASCENDING  */
    DESCENDING = 271,              /* DESCENDING  */
    SELECT = 272,                  /* SELECT  */
    GRP = 273,                     /* GRP  */
    BY = 274,                      /* BY  */
    OPEN_PAREN = 275,              /* OPEN_PAREN  */
    CLOSE_PAREN = 276,             /* CLOSE_PAREN  */
    OPEN_BRACE = 277,              /* OPEN_BRACE  */
    CLOSE_BRACE = 278,             /* CLOSE_BRACE  */
    IDENTIFIER = 279,              /* IDENTIFIER  */
    INTEGER = 280,                 /* INTEGER  */
    STRING = 281,                  /* STRING  */
    DOT = 282,                     /* DOT  */
    INVALID_STRING = 283,          /* INVALID_STRING  */
    LOWER_THAN_INTO = 284,         /* LOWER_THAN_INTO  */
    INTO = 285,                    /* INTO  */
    OR = 286,                      /* OR  */
    AND = 287,                     /* AND  */
    WITHIN = 288,                  /* WITHIN  */
    NOT = 289,                     /* NOT  */
    REL_OP = 290                   /* REL_OP  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 21 "l2h.y"

	cond_op_t relational_op;
	ordering_t ordering;
	long long number;
	char* string;
	type_info_t* type;
	fend_node_t* node;

#line 115 "l2h.tab.h"

};
typedef union YYSTYPE YYSTYPE;
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
