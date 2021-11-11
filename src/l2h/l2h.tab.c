/* A Bison parser, made by GNU Bison 3.7.4.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30704

/* Bison version string.  */
#define YYBISON_VERSION "3.7.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 1 "l2h.y"

	#include "l2h.tab.h"

    extern char *yytext;
	extern int fend_error_count;
	void yyerror(char *s, ...);
	void lyyerror(YYLTYPE t, char *s, ...);
	int yylex();

#line 81 "l2h.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "l2h.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_COMMENT = 3,                    /* COMMENT  */
  YYSYMBOL_SEMICOLON = 4,                  /* SEMICOLON  */
  YYSYMBOL_FROM = 5,                       /* FROM  */
  YYSYMBOL_TYPE = 6,                       /* TYPE  */
  YYSYMBOL_LET = 7,                        /* LET  */
  YYSYMBOL_ASSIGN = 8,                     /* ASSIGN  */
  YYSYMBOL_WHERE = 9,                      /* WHERE  */
  YYSYMBOL_ON = 10,                        /* ON  */
  YYSYMBOL_EQUALS = 11,                    /* EQUALS  */
  YYSYMBOL_JOIN = 12,                      /* JOIN  */
  YYSYMBOL_ORDERBY = 13,                   /* ORDERBY  */
  YYSYMBOL_COMMA = 14,                     /* COMMA  */
  YYSYMBOL_ASCENDING = 15,                 /* ASCENDING  */
  YYSYMBOL_DESCENDING = 16,                /* DESCENDING  */
  YYSYMBOL_SELECT = 17,                    /* SELECT  */
  YYSYMBOL_GRP = 18,                       /* GRP  */
  YYSYMBOL_BY = 19,                        /* BY  */
  YYSYMBOL_OPEN_PAREN = 20,                /* OPEN_PAREN  */
  YYSYMBOL_CLOSE_PAREN = 21,               /* CLOSE_PAREN  */
  YYSYMBOL_OPEN_BRACE = 22,                /* OPEN_BRACE  */
  YYSYMBOL_CLOSE_BRACE = 23,               /* CLOSE_BRACE  */
  YYSYMBOL_IDENTIFIER = 24,                /* IDENTIFIER  */
  YYSYMBOL_INTEGER = 25,                   /* INTEGER  */
  YYSYMBOL_STRING = 26,                    /* STRING  */
  YYSYMBOL_DOT = 27,                       /* DOT  */
  YYSYMBOL_INVALID_STRING = 28,            /* INVALID_STRING  */
  YYSYMBOL_LOWER_THAN_INTO = 29,           /* LOWER_THAN_INTO  */
  YYSYMBOL_INTO = 30,                      /* INTO  */
  YYSYMBOL_OR = 31,                        /* OR  */
  YYSYMBOL_AND = 32,                       /* AND  */
  YYSYMBOL_WITHIN = 33,                    /* WITHIN  */
  YYSYMBOL_NOT = 34,                       /* NOT  */
  YYSYMBOL_REL_OP = 35,                    /* REL_OP  */
  YYSYMBOL_YYACCEPT = 36,                  /* $accept  */
  YYSYMBOL_translation_unit = 37,          /* translation_unit  */
  YYSYMBOL_expressions = 38,               /* expressions  */
  YYSYMBOL_query = 39,                     /* query  */
  YYSYMBOL_40_1 = 40,                      /* $@1  */
  YYSYMBOL_comment = 41,                   /* comment  */
  YYSYMBOL_query_expression = 42,          /* query_expression  */
  YYSYMBOL_query_body = 43,                /* query_body  */
  YYSYMBOL_opt_query_body_clauses = 44,    /* opt_query_body_clauses  */
  YYSYMBOL_query_continuation = 45,        /* query_continuation  */
  YYSYMBOL_46_2 = 46,                      /* $@2  */
  YYSYMBOL_query_body_clauses = 47,        /* query_body_clauses  */
  YYSYMBOL_query_body_clause = 48,         /* query_body_clause  */
  YYSYMBOL_from_clause = 49,               /* from_clause  */
  YYSYMBOL_let_clause = 50,                /* let_clause  */
  YYSYMBOL_where_clause = 51,              /* where_clause  */
  YYSYMBOL_join_clause = 52,               /* join_clause  */
  YYSYMBOL_join_into_clause = 53,          /* join_into_clause  */
  YYSYMBOL_orderby_clause = 54,            /* orderby_clause  */
  YYSYMBOL_orderings = 55,                 /* orderings  */
  YYSYMBOL_ordering = 56,                  /* ordering  */
  YYSYMBOL_ordering_direction = 57,        /* ordering_direction  */
  YYSYMBOL_select_or_group_clause = 58,    /* select_or_group_clause  */
  YYSYMBOL_select_clause = 59,             /* select_clause  */
  YYSYMBOL_group_clause = 60,              /* group_clause  */
  YYSYMBOL_boolean_expression = 61,        /* boolean_expression  */
  YYSYMBOL_conditional_or_expression = 62, /* conditional_or_expression  */
  YYSYMBOL_conditional_and_expression = 63, /* conditional_and_expression  */
  YYSYMBOL_not_expression = 64,            /* not_expression  */
  YYSYMBOL_exclusive_or_expression = 65,   /* exclusive_or_expression  */
  YYSYMBOL_expression = 66,                /* expression  */
  YYSYMBOL_unary_expression = 67,          /* unary_expression  */
  YYSYMBOL_anonymous_object = 68,          /* anonymous_object  */
  YYSYMBOL_anonymous_object_initialization = 69, /* anonymous_object_initialization  */
  YYSYMBOL_relational_expr = 70,           /* relational_expr  */
  YYSYMBOL_identifier = 71,                /* identifier  */
  YYSYMBOL_attribute = 72,                 /* attribute  */
  YYSYMBOL_invocation_expression = 73,     /* invocation_expression  */
  YYSYMBOL_opt_argument_list = 74,         /* opt_argument_list  */
  YYSYMBOL_argument_list = 75,             /* argument_list  */
  YYSYMBOL_typedef = 76,                   /* typedef  */
  YYSYMBOL_type = 77                       /* type  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if 1

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* 1 */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
             && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE) \
             + YYSIZEOF (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  7
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   102

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  36
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  42
/* YYNRULES -- Number of rules.  */
#define YYNRULES  72
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  113

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   290


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   108,   108,   111,   112,   116,   117,   117,   121,   125,
     129,   133,   134,   138,   138,   139,   143,   144,   148,   149,
     150,   151,   152,   153,   157,   161,   165,   169,   173,   177,
     181,   182,   186,   190,   191,   192,   196,   197,   201,   205,
     209,   213,   214,   218,   219,   223,   224,   228,   229,   233,
     234,   235,   239,   240,   241,   242,   243,   247,   251,   252,
     256,   260,   264,   265,   269,   273,   274,   278,   279,   283,
     284,   288,   289
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if 1
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "COMMENT", "SEMICOLON",
  "FROM", "TYPE", "LET", "ASSIGN", "WHERE", "ON", "EQUALS", "JOIN",
  "ORDERBY", "COMMA", "ASCENDING", "DESCENDING", "SELECT", "GRP", "BY",
  "OPEN_PAREN", "CLOSE_PAREN", "OPEN_BRACE", "CLOSE_BRACE", "IDENTIFIER",
  "INTEGER", "STRING", "DOT", "INVALID_STRING", "LOWER_THAN_INTO", "INTO",
  "OR", "AND", "WITHIN", "NOT", "REL_OP", "$accept", "translation_unit",
  "expressions", "query", "$@1", "comment", "query_expression",
  "query_body", "opt_query_body_clauses", "query_continuation", "$@2",
  "query_body_clauses", "query_body_clause", "from_clause", "let_clause",
  "where_clause", "join_clause", "join_into_clause", "orderby_clause",
  "orderings", "ordering", "ordering_direction", "select_or_group_clause",
  "select_clause", "group_clause", "boolean_expression",
  "conditional_or_expression", "conditional_and_expression",
  "not_expression", "exclusive_or_expression", "expression",
  "unary_expression", "anonymous_object",
  "anonymous_object_initialization", "relational_expr", "identifier",
  "attribute", "invocation_expression", "opt_argument_list",
  "argument_list", "typedef", "type", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290
};
#endif

#define YYPACT_NINF (-31)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-73)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int8 yypact[] =
{
      10,   -31,    17,     3,   -31,    14,   -31,   -31,   -31,    -1,
      16,    49,   -31,     8,   -31,    24,    12,   -31,    12,    21,
      -1,    13,   -31,    -2,    49,   -31,   -31,   -31,   -31,   -31,
     -31,   -31,    13,   -31,   -31,    42,    21,   -31,   -31,     5,
     -31,    28,    31,   -31,   -31,   -13,   -31,    38,    27,    18,
     -31,    54,   -31,    33,   -31,   -31,    13,    13,    39,   -31,
     -31,   -31,   -31,    13,    50,   -31,    21,    21,    18,     4,
      13,   -31,   -11,    13,   -31,   -31,   -31,   -31,    53,    12,
     -31,   -31,   -31,    31,   -31,   -31,   -31,    55,   -31,   -31,
      63,    18,   -31,   -31,    13,   -31,    13,    13,   -31,   -31,
      49,   -31,    58,    60,    69,   -31,   -31,    13,    13,   -31,
      51,    12,   -31
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       6,     8,     0,     2,     3,     0,     5,     1,     4,     0,
       0,    11,    71,    61,    70,     0,     0,     7,     0,     0,
       0,     0,     9,     0,    12,    16,    18,    19,    20,    21,
      22,    23,     0,    61,    69,     0,     0,    56,    55,     0,
      26,    40,    41,    43,    45,     0,    48,    52,     0,     0,
      51,    29,    30,    33,    49,    50,     0,     0,    15,    36,
      37,    17,    24,     0,     0,    46,     0,     0,     0,     0,
       0,    58,     0,     0,    34,    35,    32,    38,     0,     0,
      10,    25,    47,    42,    44,    60,    63,    62,    53,    54,
       0,     0,    57,    31,     0,    13,    66,     0,    59,    39,
      11,    67,     0,    65,     0,    14,    64,     0,     0,    68,
      27,     0,    28
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -31,   -31,   -31,    79,   -31,   -31,    78,   -16,   -31,   -31,
     -31,   -31,    61,   -10,   -31,   -31,   -31,   -31,   -31,   -31,
      15,   -31,   -31,   -31,   -31,    56,   -31,    20,    22,    48,
     -30,   -15,   -31,   -31,   -31,    -9,   -31,   -31,   -31,   -31,
      71,   -31
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     2,     3,     4,     5,     6,    50,    22,    23,    80,
     100,    24,    25,    11,    27,    28,    29,    30,    31,    51,
      52,    76,    58,    59,    60,    40,    41,    42,    43,    44,
      53,    54,    55,    72,    46,    47,    88,    89,   102,   103,
      15,    16
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
{
      14,    26,    62,    91,    45,    12,     1,    34,    -6,    35,
      86,    14,    92,     1,    26,    56,    57,     7,     9,     9,
      17,    45,    68,    13,    45,    36,    77,    78,    87,    33,
      37,    38,   -72,    81,    71,    49,    33,    33,    37,    38,
      90,    36,    33,    37,    38,    33,    37,    38,    74,    75,
      63,    45,    45,    85,     9,    39,    18,    32,    19,    66,
      70,    20,    21,    67,    99,    69,   101,   104,    73,    79,
      95,    82,    94,    97,   107,    96,    98,   109,   110,   106,
     108,   111,     8,    10,   105,    61,    83,    65,    93,    84,
      26,    48,    64,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   112
};

static const yytype_int8 yycheck[] =
{
       9,    11,    32,    14,    19,     6,     3,    16,     5,    18,
       6,    20,    23,     3,    24,    17,    18,     0,     5,     5,
       4,    36,    35,    24,    39,    20,    56,    57,    24,    24,
      25,    26,    24,    63,    49,    22,    24,    24,    25,    26,
      70,    20,    24,    25,    26,    24,    25,    26,    15,    16,
       8,    66,    67,    68,     5,    34,     7,    33,     9,    31,
      33,    12,    13,    32,    94,    27,    96,    97,    14,    30,
      79,    21,    19,    10,    14,    20,    91,   107,   108,    21,
      11,    30,     3,     5,   100,    24,    66,    39,    73,    67,
     100,    20,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   111
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,     3,    37,    38,    39,    40,    41,     0,    39,     5,
      42,    49,     6,    24,    71,    76,    77,     4,     7,     9,
      12,    13,    43,    44,    47,    48,    49,    50,    51,    52,
      53,    54,    33,    24,    71,    71,    20,    25,    26,    34,
      61,    62,    63,    64,    65,    67,    70,    71,    76,    22,
      42,    55,    56,    66,    67,    68,    17,    18,    58,    59,
      60,    48,    66,     8,    61,    65,    31,    32,    35,    27,
      33,    67,    69,    14,    15,    16,    57,    66,    66,    30,
      45,    66,    21,    63,    64,    67,     6,    24,    72,    73,
      66,    14,    23,    56,    19,    71,    20,    10,    67,    66,
      46,    66,    74,    75,    66,    43,    21,    14,    11,    66,
      66,    30,    71
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] =
{
       0,    36,    37,    38,    38,    39,    40,    39,    41,    42,
      43,    44,    44,    46,    45,    45,    47,    47,    48,    48,
      48,    48,    48,    48,    49,    50,    51,    52,    53,    54,
      55,    55,    56,    57,    57,    57,    58,    58,    59,    60,
      61,    62,    62,    63,    63,    64,    64,    65,    65,    66,
      66,    66,    67,    67,    67,    67,    67,    68,    69,    69,
      70,    71,    72,    72,    73,    74,    74,    75,    75,    76,
      76,    77,    77
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     1,     2,     1,     0,     3,     1,     2,
       3,     0,     1,     0,     4,     0,     1,     2,     1,     1,
       1,     1,     1,     1,     4,     4,     2,     8,    10,     2,
       1,     3,     2,     0,     1,     1,     1,     1,     2,     4,
       1,     1,     3,     1,     3,     1,     2,     3,     1,     1,
       1,     1,     1,     3,     3,     1,     1,     3,     1,     3,
       3,     1,     1,     1,     4,     1,     0,     1,     3,     2,
       1,     1,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

# ifndef YY_LOCATION_PRINT
#  if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static int
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  int res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
 }

#   define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

#  else
#   define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#  endif
# endif /* !defined YY_LOCATION_PRINT */


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  YYUSE (yylocationp);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yykind < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yykind], *yyvaluep);
# endif
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  YY_LOCATION_PRINT (yyo, *yylocationp);
  YYFPRINTF (yyo, ": ");
  yy_symbol_value_print (yyo, yykind, yyvaluep, yylocationp);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)],
                       &(yylsp[(yyi + 1) - (yynrhs)]));
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


/* Context of a parse error.  */
typedef struct
{
  yy_state_t *yyssp;
  yysymbol_kind_t yytoken;
  YYLTYPE *yylloc;
} yypcontext_t;

/* Put in YYARG at most YYARGN of the expected tokens given the
   current YYCTX, and return the number of tokens stored in YYARG.  If
   YYARG is null, return the number of expected tokens (guaranteed to
   be less than YYNTOKENS).  Return YYENOMEM on memory exhaustion.
   Return 0 if there are more than YYARGN expected tokens, yet fill
   YYARG up to YYARGN. */
static int
yypcontext_expected_tokens (const yypcontext_t *yyctx,
                            yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  int yyn = yypact[+*yyctx->yyssp];
  if (!yypact_value_is_default (yyn))
    {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yyx;
      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck[yyx + yyn] == yyx && yyx != YYSYMBOL_YYerror
            && !yytable_value_is_error (yytable[yyx + yyn]))
          {
            if (!yyarg)
              ++yycount;
            else if (yycount == yyargn)
              return 0;
            else
              yyarg[yycount++] = YY_CAST (yysymbol_kind_t, yyx);
          }
    }
  if (yyarg && yycount == 0 && 0 < yyargn)
    yyarg[0] = YYSYMBOL_YYEMPTY;
  return yycount;
}




#ifndef yystrlen
# if defined __GLIBC__ && defined _STRING_H
#  define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
# else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
# endif
#endif

#ifndef yystpcpy
# if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#  define yystpcpy stpcpy
# else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
# endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
      char const *yyp = yystr;
      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            else
              goto append;

          append:
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
}
#endif


static int
yy_syntax_error_arguments (const yypcontext_t *yyctx,
                           yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yyctx->yytoken != YYSYMBOL_YYEMPTY)
    {
      int yyn;
      if (yyarg)
        yyarg[yycount] = yyctx->yytoken;
      ++yycount;
      yyn = yypcontext_expected_tokens (yyctx,
                                        yyarg ? yyarg + 1 : yyarg, yyargn - 1);
      if (yyn == YYENOMEM)
        return YYENOMEM;
      else
        yycount += yyn;
    }
  return yycount;
}

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return -1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return YYENOMEM if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                const yypcontext_t *yyctx)
{
  enum { YYARGS_MAX = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  yysymbol_kind_t yyarg[YYARGS_MAX];
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

  /* Actual size of YYARG. */
  int yycount = yy_syntax_error_arguments (yyctx, yyarg, YYARGS_MAX);
  if (yycount == YYENOMEM)
    return YYENOMEM;

  switch (yycount)
    {
#define YYCASE_(N, S)                       \
      case N:                               \
        yyformat = S;                       \
        break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
    }

  /* Compute error message size.  Don't count the "%s"s, but reserve
     room for the terminator.  */
  yysize = yystrlen (yyformat) - 2 * yycount + 1;
  {
    int yyi;
    for (yyi = 0; yyi < yycount; ++yyi)
      {
        YYPTRDIFF_T yysize1
          = yysize + yytnamerr (YY_NULLPTR, yytname[yyarg[yyi]]);
        if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
          yysize = yysize1;
        else
          return YYENOMEM;
      }
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return -1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yytname[yyarg[yyi++]]);
          yyformat += 2;
        }
      else
        {
          ++yyp;
          ++yyformat;
        }
  }
  return 0;
}


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

    /* The location stack: array, bottom, top.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls = yylsa;
    YYLTYPE *yylsp = yyls;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[3];

  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */
  yylsp[0] = yylloc;
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yyls1, yysize * YYSIZEOF (*yylsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
        yyls = yyls1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      yyerror_range[1] = yylloc;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location. */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  yyerror_range[1] = yyloc;
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 6: /* $@1: %empty  */
#line 117 "l2h.y"
          { fend_query_init(); }
#line 1600 "l2h.tab.c"
    break;

  case 7: /* query: $@1 query_expression SEMICOLON  */
#line 117 "l2h.y"
                                                            { fend_query_cleanup((yyvsp[-1].node)); }
#line 1606 "l2h.tab.c"
    break;

  case 9: /* query_expression: from_clause query_body  */
#line 125 "l2h.y"
                                 { (yyval.node) = fend_query_complete((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1612 "l2h.tab.c"
    break;

  case 10: /* query_body: opt_query_body_clauses select_or_group_clause query_continuation  */
#line 129 "l2h.y"
                                                                           { (yyval.node) = fend_on_query_body((yyvsp[-2].node), (yyvsp[-1].node), (yyvsp[0].node)); }
#line 1618 "l2h.tab.c"
    break;

  case 11: /* opt_query_body_clauses: %empty  */
#line 133 "l2h.y"
          { (yyval.node) = NULL; }
#line 1624 "l2h.tab.c"
    break;

  case 12: /* opt_query_body_clauses: query_body_clauses  */
#line 134 "l2h.y"
                             { (yyval.node) = (yyvsp[0].node); }
#line 1630 "l2h.tab.c"
    break;

  case 13: /* $@2: %empty  */
#line 138 "l2h.y"
                          { fend_register_identifier((yyvsp[0].node), type_def_user); }
#line 1636 "l2h.tab.c"
    break;

  case 14: /* query_continuation: INTO identifier $@2 query_body  */
#line 138 "l2h.y"
                                                                                      { (yyval.node) = fend_on_continuation((yyvsp[-2].node), (yyvsp[0].node)); }
#line 1642 "l2h.tab.c"
    break;

  case 15: /* query_continuation: %empty  */
#line 139 "l2h.y"
          { (yyval.node) = NULL; }
#line 1648 "l2h.tab.c"
    break;

  case 16: /* query_body_clauses: query_body_clause  */
#line 143 "l2h.y"
                            { (yyval.node) = (yyvsp[0].node); }
#line 1654 "l2h.tab.c"
    break;

  case 17: /* query_body_clauses: query_body_clauses query_body_clause  */
#line 144 "l2h.y"
                                               { (yyval.node) = fend_on_enum((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1660 "l2h.tab.c"
    break;

  case 24: /* from_clause: FROM typedef WITHIN expression  */
#line 157 "l2h.y"
                                         { (yyval.node) = fend_on_from((yyvsp[-2].node), (yyvsp[0].node)); }
#line 1666 "l2h.tab.c"
    break;

  case 25: /* let_clause: LET identifier ASSIGN expression  */
#line 161 "l2h.y"
                                           { (yyval.node) = fend_on_let((yyvsp[-2].node), (yyvsp[0].node)); }
#line 1672 "l2h.tab.c"
    break;

  case 26: /* where_clause: WHERE boolean_expression  */
#line 165 "l2h.y"
                                   { (yyval.node) = fend_on_where((yyvsp[0].node)); }
#line 1678 "l2h.tab.c"
    break;

  case 27: /* join_clause: JOIN typedef WITHIN expression ON expression EQUALS expression  */
#line 169 "l2h.y"
                                                                         { (yyval.node) = fend_on_join((yyvsp[-6].node), (yyvsp[-4].node), (yyvsp[-2].node), (yyvsp[0].node)); }
#line 1684 "l2h.tab.c"
    break;

  case 28: /* join_into_clause: JOIN typedef WITHIN expression ON expression EQUALS expression INTO identifier  */
#line 173 "l2h.y"
                                                                                         { fend_register_identifier((yyvsp[0].node), type_def_user); (yyval.node) = fend_on_continuation((yyvsp[0].node), fend_on_join((yyvsp[-8].node), (yyvsp[-6].node), (yyvsp[-4].node), (yyvsp[-2].node))); }
#line 1690 "l2h.tab.c"
    break;

  case 29: /* orderby_clause: ORDERBY orderings  */
#line 177 "l2h.y"
                            { (yyval.node) = fend_on_order_by((yyvsp[0].node)); }
#line 1696 "l2h.tab.c"
    break;

  case 30: /* orderings: ordering  */
#line 181 "l2h.y"
                   { (yyval.node) = (yyvsp[0].node); }
#line 1702 "l2h.tab.c"
    break;

  case 31: /* orderings: orderings COMMA ordering  */
#line 182 "l2h.y"
                                   { (yyval.node) = fend_on_enum((yyvsp[-2].node), (yyvsp[0].node)); }
#line 1708 "l2h.tab.c"
    break;

  case 32: /* ordering: expression ordering_direction  */
#line 186 "l2h.y"
                                        { (yyval.node) = fend_on_ordering((yyvsp[-1].node), (yyvsp[0].ordering)); }
#line 1714 "l2h.tab.c"
    break;

  case 33: /* ordering_direction: %empty  */
#line 190 "l2h.y"
          { (yyval.ordering) = ordering_asc; }
#line 1720 "l2h.tab.c"
    break;

  case 34: /* ordering_direction: ASCENDING  */
#line 191 "l2h.y"
                    { (yyval.ordering) = (yyvsp[0].ordering); }
#line 1726 "l2h.tab.c"
    break;

  case 35: /* ordering_direction: DESCENDING  */
#line 192 "l2h.y"
                     { (yyval.ordering) = (yyvsp[0].ordering); }
#line 1732 "l2h.tab.c"
    break;

  case 38: /* select_clause: SELECT expression  */
#line 201 "l2h.y"
                            { (yyval.node) = (yyvsp[0].node); }
#line 1738 "l2h.tab.c"
    break;

  case 39: /* group_clause: GRP expression BY expression  */
#line 205 "l2h.y"
                                       { (yyval.node) = fend_on_group((yyvsp[-2].node), (yyvsp[0].node)); }
#line 1744 "l2h.tab.c"
    break;

  case 41: /* conditional_or_expression: conditional_and_expression  */
#line 213 "l2h.y"
                                     { (yyval.node) = (yyvsp[0].node); }
#line 1750 "l2h.tab.c"
    break;

  case 42: /* conditional_or_expression: conditional_or_expression OR conditional_and_expression  */
#line 214 "l2h.y"
                                                                  { (yyval.node) = fend_on_predicate((yyvsp[-2].node), (yyvsp[0].node), node_type_or_rel); }
#line 1756 "l2h.tab.c"
    break;

  case 43: /* conditional_and_expression: not_expression  */
#line 218 "l2h.y"
                         { (yyval.node) = (yyvsp[0].node); }
#line 1762 "l2h.tab.c"
    break;

  case 44: /* conditional_and_expression: conditional_and_expression AND not_expression  */
#line 219 "l2h.y"
                                                        { (yyval.node) = fend_on_predicate((yyvsp[-2].node), (yyvsp[0].node), node_type_and_rel); }
#line 1768 "l2h.tab.c"
    break;

  case 45: /* not_expression: exclusive_or_expression  */
#line 223 "l2h.y"
                                  { (yyval.node) = (yyvsp[0].node); }
#line 1774 "l2h.tab.c"
    break;

  case 46: /* not_expression: NOT exclusive_or_expression  */
#line 224 "l2h.y"
                                      { (yyval.node) = fend_on_predicate((yyvsp[0].node), NULL, node_type_not_rel); }
#line 1780 "l2h.tab.c"
    break;

  case 47: /* exclusive_or_expression: OPEN_PAREN boolean_expression CLOSE_PAREN  */
#line 228 "l2h.y"
                                                    { (yyval.node) = (yyvsp[-1].node); }
#line 1786 "l2h.tab.c"
    break;

  case 48: /* exclusive_or_expression: relational_expr  */
#line 229 "l2h.y"
                          { (yyval.node) = (yyvsp[0].node); }
#line 1792 "l2h.tab.c"
    break;

  case 52: /* unary_expression: identifier  */
#line 239 "l2h.y"
                     { (yyval.node) = fend_on_unary_expression(unary_exp_type_identifier, (yyvsp[0].node), NULL); }
#line 1798 "l2h.tab.c"
    break;

  case 53: /* unary_expression: identifier DOT attribute  */
#line 240 "l2h.y"
                                   { if (!fend_is_identifier_defined((yyvsp[-2].node))) lyyerror((yylsp[-2]),"identifier %s undefined", (yyvsp[-2].node)->value.string); (yyval.node) = fend_on_unary_expression(unary_exp_type_property_call, (yyvsp[-2].node), (yyvsp[0].node)); }
#line 1804 "l2h.tab.c"
    break;

  case 54: /* unary_expression: identifier DOT invocation_expression  */
#line 241 "l2h.y"
                                               { if (!fend_is_identifier_defined((yyvsp[-2].node))) lyyerror((yylsp[-2]),"identifier %s undefined", (yyvsp[-2].node)->value.string); (yyval.node) = fend_on_unary_expression(unary_exp_type_mehtod_call, (yyvsp[-2].node), (yyvsp[0].node)); }
#line 1810 "l2h.tab.c"
    break;

  case 55: /* unary_expression: STRING  */
#line 242 "l2h.y"
                 { (yyval.node) = fend_on_unary_expression(unary_exp_type_string, (yyvsp[0].string), NULL); }
#line 1816 "l2h.tab.c"
    break;

  case 56: /* unary_expression: INTEGER  */
#line 243 "l2h.y"
                  { (yyval.node) = fend_on_unary_expression(unary_exp_type_number, (yyvsp[0].number), NULL); }
#line 1822 "l2h.tab.c"
    break;

  case 57: /* anonymous_object: OPEN_BRACE anonymous_object_initialization CLOSE_BRACE  */
#line 247 "l2h.y"
                                                                 { (yyval.node) = (yyvsp[-1].node); }
#line 1828 "l2h.tab.c"
    break;

  case 58: /* anonymous_object_initialization: unary_expression  */
#line 251 "l2h.y"
                           { (yyval.node) = (yyvsp[0].node); }
#line 1834 "l2h.tab.c"
    break;

  case 59: /* anonymous_object_initialization: anonymous_object_initialization COMMA unary_expression  */
#line 252 "l2h.y"
                                                                 { (yyval.node) = fend_on_enum((yyvsp[-2].node), (yyvsp[0].node)); }
#line 1840 "l2h.tab.c"
    break;

  case 60: /* relational_expr: unary_expression REL_OP unary_expression  */
#line 256 "l2h.y"
                                                   { (yyval.node) = fend_on_releational_expr((yyvsp[-2].node), (yyvsp[0].node), (yyvsp[-1].relational_op)); }
#line 1846 "l2h.tab.c"
    break;

  case 61: /* identifier: IDENTIFIER  */
#line 260 "l2h.y"
                     { (yyval.node) = fend_on_identifier((yyvsp[0].string)); }
#line 1852 "l2h.tab.c"
    break;

  case 62: /* attribute: IDENTIFIER  */
#line 264 "l2h.y"
                     { (yyval.node) = fend_on_string_attribute((yyvsp[0].string)); }
#line 1858 "l2h.tab.c"
    break;

  case 63: /* attribute: TYPE  */
#line 265 "l2h.y"
               { (yyval.node) = fend_on_type_attribute((yyvsp[0].type)); }
#line 1864 "l2h.tab.c"
    break;

  case 64: /* invocation_expression: IDENTIFIER OPEN_PAREN opt_argument_list CLOSE_PAREN  */
#line 269 "l2h.y"
                                                              {(yyval.node) = fend_on_method_call((yyvsp[-3].string), (yyvsp[-1].node)); }
#line 1870 "l2h.tab.c"
    break;

  case 65: /* opt_argument_list: argument_list  */
#line 273 "l2h.y"
                        { (yyval.node) = (yyvsp[0].node); }
#line 1876 "l2h.tab.c"
    break;

  case 66: /* opt_argument_list: %empty  */
#line 274 "l2h.y"
          { (yyval.node) = NULL; }
#line 1882 "l2h.tab.c"
    break;

  case 67: /* argument_list: expression  */
#line 278 "l2h.y"
                     { (yyval.node) = (yyvsp[0].node); }
#line 1888 "l2h.tab.c"
    break;

  case 68: /* argument_list: argument_list COMMA expression  */
#line 279 "l2h.y"
                                         { (yyval.node) = fend_on_enum((yyvsp[-2].node), (yyvsp[0].node)); }
#line 1894 "l2h.tab.c"
    break;

  case 69: /* typedef: type identifier  */
#line 283 "l2h.y"
                      { (yyval.node) = fend_on_identifier_declaration((yyvsp[-1].type), (yyvsp[0].node)); }
#line 1900 "l2h.tab.c"
    break;

  case 70: /* typedef: identifier  */
#line 284 "l2h.y"
                     { (yyval.node) = fend_on_identifier_declaration(fend_on_simple_type_def(type_def_dynamic), (yyvsp[0].node)); }
#line 1906 "l2h.tab.c"
    break;

  case 71: /* type: TYPE  */
#line 288 "l2h.y"
               { (yyval.type) = (yyvsp[0].type); }
#line 1912 "l2h.tab.c"
    break;

  case 72: /* type: IDENTIFIER  */
#line 289 "l2h.y"
                     { (yyval.type) = fend_on_complex_type_def(type_def_user, (yyvsp[0].string)); }
#line 1918 "l2h.tab.c"
    break;


#line 1922 "l2h.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      {
        yypcontext_t yyctx
          = {yyssp, yytoken, &yylloc};
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == -1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = YY_CAST (char *,
                             YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
            if (yymsg)
              {
                yysyntax_error_status
                  = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
                yymsgp = yymsg;
              }
            else
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = YYENOMEM;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == YYENOMEM)
          goto yyexhaustedlab;
      }
    }

  yyerror_range[1] = yylloc;
  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  ++yylsp;
  YYLLOC_DEFAULT (*yylsp, yyerror_range, 2);

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;


#if 1
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturn;
#endif


/*-------------------------------------------------------.
| yyreturn -- parsing is finished, clean up and return.  |
`-------------------------------------------------------*/
yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
  return yyresult;
}

#line 292 "l2h.y"


void yyerror(char *s, ...)
{
	va_list ap;
	va_start(ap, s);
	if(yylloc.first_line)
		lib_fprintf(stderr, "%d.%d-%d.%d: error: ", yylloc.first_line, yylloc.first_column, yylloc.last_line, yylloc.last_column);
#ifdef __STDC_WANT_SECURE_LIB__
    vfprintf_s(stderr, s, ap);
#else
    vfprintf(stderr, s, ap);
#endif
	va_end(ap);
	lib_fprintf(stderr, "\n");
	fend_error_count++;
	fend_query_cleanup(NULL);
}

void lyyerror(YYLTYPE t, char *s, ...)
{
	va_list ap;
	va_start(ap, s);
	if(t.first_line)
		lib_fprintf(stderr, "%d.%d-%d.%d: error: ", t.first_line, t.first_column, t.last_line, t.last_column);
#ifdef __STDC_WANT_SECURE_LIB__
    vfprintf_s(stderr, s, ap);
#else
    vfprintf(stderr, s, ap);
#endif
	va_end(ap);
	fend_error_count++;
	lib_fprintf(stderr, "\n");
}
