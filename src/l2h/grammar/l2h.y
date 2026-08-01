%{
	#include "l2h.tab.h"

	extern int yylineno;
    extern char *yytext;
	extern int fend_error_count;
	void yyerror(char *s, ...);
	void lyyerror(YYLTYPE t, char *s, ...);
	int yylex();

	#define FLOC(n, L) do { \
		if ((n) != NULL) \
			fend_node_set_loc((n), (L).first_line, (L).first_column, (L).last_line, (L).last_column); \
	} while (0)
%}

%define parse.error verbose

%code requires
{
	#include <stdarg.h>
	#include <stdio.h>
	#include "lib.h"
	#include "frontend.h"
}

%locations

%union {
	cond_op_t relational_op;
	ordering_t ordering;
	long long number;
	char* string;
	type_info_t* type;
	fend_node_t* node;
}

%start translation_unit

%token COMMENT
%token SEMICOLON
%token FROM
%token <type> TYPE
%token LET
%token ASSIGN
%token WHERE
%token ON
%token EQUALS
%token JOIN
%token ORDERBY
%token COMMA
%token <ordering> ASCENDING
%token <ordering> DESCENDING
%token SELECT
%token GRP
%token BY
%token OPEN_PAREN
%token CLOSE_PAREN
%token OPEN_BRACE
%token CLOSE_BRACE
%token <string> IDENTIFIER

%token <number> INTEGER
%token <string> STRING
%token BOOL_TRUE
%token BOOL_FALSE
%token DOT
%token <string> INVALID_STRING

/* operators and precedence levels */

%nonassoc LOWER_THAN_INTO
%nonassoc INTO
%left OR
%left AND
%nonassoc WITHIN
%left NOT
%left <relational_op> REL_OP

%type <type> type
%type <node> typedef
%type <node> attribute
%type <node> query_expression
%type <node> query_expression_nested
%type <node> query_body_nested
%type <node> expression
%type <node> value_expression
%type <node> unary_expression
%type <node> relational_expr
%type <node> boolean_expression
%type <node> conditional_or_expression
%type <node> conditional_and_expression
%type <node> not_expression
%type <node> exclusive_or_expression
%type <node> anonymous_object
%type <node> anonymous_object_initialization
%type <node> anonymous_object_field
%type <node> select_clause
%type <node> group_clause
%type <node> select_or_group_clause
%type <node> from_clause
%type <node> where_clause
%type <node> query_body_clauses
%type <node> opt_query_body_clauses
%type <node> query_body_clause
%type <node> let_clause
%type <node> query_body
%type <node> query_continuation
%type <node> invocation_expression
%type <node> opt_argument_list
%type <node> argument_list
%type <node> join_clause
%type <node> join_into_clause
%type <node> identifier
%type <node> orderby_clause
%type <node> orderings
%type <node> ordering
%type <ordering> ordering_direction

%%

translation_unit : expressions ;

expressions 
    : query 
    | expressions query
    ;

query
	: comment
	| { fend_query_init(); } query_expression SEMICOLON { fend_query_cleanup($2); }
	;

comment 
	: COMMENT
	;
	
query_expression
	: from_clause query_body { $$ = fend_query_complete($1, $2); FLOC($$, @$); }
	;

/* Nested query as a value: no `into` continuation, so outer `into` binds correctly. */
query_expression_nested
	: from_clause query_body_nested { $$ = fend_query_complete($1, $2); FLOC($$, @$); }
	;

query_body
	: opt_query_body_clauses select_or_group_clause query_continuation { $$ = fend_on_query_body($1, $2, $3); FLOC($$, @$); }
	;

query_body_nested
	: opt_query_body_clauses select_or_group_clause { $$ = fend_on_query_body($1, $2, NULL); FLOC($$, @$); }
	;

opt_query_body_clauses
	: { $$ = NULL; }
	| query_body_clauses { $$ = $1; }
	;

query_continuation
	: INTO identifier { fend_register_identifier($2); } query_body { $$ = fend_on_continuation($2, $4); FLOC($$, @$); }
	| { $$ = NULL; } %prec LOWER_THAN_INTO 
	;

query_body_clauses
	: query_body_clause { $$ = $1; }
	| query_body_clauses query_body_clause { $$ = fend_on_enum($1, $2); FLOC($$, @$); }
	;

query_body_clause
	: from_clause
	| let_clause
    | where_clause
    | join_clause
    | join_into_clause
    | orderby_clause
	;

from_clause
	: FROM typedef WITHIN value_expression { $$ = fend_on_from($2, $4); FLOC($$, @$); }
	;

let_clause
	: LET identifier ASSIGN value_expression { $$ = fend_on_let($2, $4); FLOC($$, @$); }
	;

where_clause
	: WHERE boolean_expression { $$ = fend_on_where($2); FLOC($$, @$); }
	;

join_clause
	: JOIN typedef WITHIN value_expression ON value_expression EQUALS value_expression { $$ = fend_on_join($2, $4, $6, $8); FLOC($$, @$); }
	;

join_into_clause
	: JOIN typedef WITHIN value_expression ON value_expression EQUALS value_expression INTO identifier {
		fend_register_identifier($10);
		fend_node_t* j = fend_on_join($2, $4, $6, $8);
		FLOC(j, @$);
		$$ = fend_on_continuation($10, j);
		FLOC($$, @$);
	}
	;

orderby_clause
	: ORDERBY orderings { $$ = fend_on_order_by($2); FLOC($$, @$); }
	;

orderings
	: ordering { $$ = $1; }
	| orderings COMMA ordering { $$ = fend_on_enum($1, $3); FLOC($$, @$); }
	;

ordering
	: value_expression ordering_direction { $$ = fend_on_ordering($1, $2); FLOC($$, @$); }
	;

ordering_direction
	: { $$ = ordering_asc; }
	| ASCENDING { $$ = $1; }
	| DESCENDING { $$ = $1; }
	;

select_or_group_clause
	: select_clause
	| group_clause
	;

select_clause
	: SELECT value_expression { $$ = $2; }
	;

group_clause
	: GRP value_expression BY value_expression { $$ = fend_on_group($2, $4); FLOC($$, @$); }
	;

boolean_expression
	: conditional_or_expression
	;

conditional_or_expression
	: conditional_and_expression { $$ = $1; }
	| conditional_or_expression OR conditional_and_expression { $$ = fend_on_predicate($1, $3, node_type_or_rel); FLOC($$, @$); }
	;

conditional_and_expression
	: not_expression { $$ = $1; }
	| conditional_and_expression AND not_expression { $$ = fend_on_predicate($1, $3, node_type_and_rel); FLOC($$, @$); }
	;

not_expression
	: exclusive_or_expression { $$ = $1; }
	| NOT exclusive_or_expression { $$ = fend_on_predicate($2, NULL, node_type_not_rel); FLOC($$, @$); }
	;

exclusive_or_expression
	: OPEN_PAREN boolean_expression CLOSE_PAREN { $$ = $2; }
	| query_expression_nested { $$ = $1; }
	| relational_expr { $$ = $1; }
	| BOOL_TRUE { $$ = fend_on_boolean_literal(1); FLOC($$, @$); }
	| BOOL_FALSE { $$ = fend_on_boolean_literal(0); FLOC($$, @$); }
	;

expression
	: unary_expression
	| anonymous_object
	;

value_expression
	: expression
	| query_expression_nested
	;
	
unary_expression
	: identifier { $$ = fend_on_unary_expression(unary_exp_type_identifier, $1, NULL); FLOC($$, @$); }
	| identifier DOT attribute { if (!fend_is_identifier_defined($1)) lyyerror(@1,"identifier %s undefined", $1->value.string); $$ = fend_on_unary_expression(unary_exp_type_property_call, $1, $3); FLOC($$, @$); }
	| identifier DOT invocation_expression { if (!fend_is_identifier_defined($1)) lyyerror(@1,"identifier %s undefined", $1->value.string); $$ = fend_on_unary_expression(unary_exp_type_mehtod_call, $1, $3); FLOC($$, @$); }
	| STRING { $$ = fend_on_unary_expression(unary_exp_type_string, $1, NULL); FLOC($$, @$); }
	| INTEGER { $$ = fend_on_unary_expression(unary_exp_type_number, (void*)$1, NULL); FLOC($$, @$); }
	| BOOL_TRUE { $$ = fend_on_boolean_literal(1); FLOC($$, @$); }
	| BOOL_FALSE { $$ = fend_on_boolean_literal(0); FLOC($$, @$); }
	;
	
anonymous_object
	: OPEN_BRACE anonymous_object_initialization CLOSE_BRACE { $$ = fend_on_object($2); FLOC($$, @$); }
	;
	
anonymous_object_initialization
	: anonymous_object_field { $$ = $1; }
	| anonymous_object_initialization COMMA anonymous_object_field { $$ = fend_on_enum($1, $3); FLOC($$, @$); }
	;

anonymous_object_field
	: unary_expression { $$ = $1; }
	| identifier ASSIGN value_expression { $$ = fend_on_named_field($1, $3); FLOC($$, @$); }
	;

relational_expr
	: value_expression REL_OP value_expression { $$ = fend_on_releational_expr($1, $3, $2); FLOC($$, @$); }
	;

identifier 
	: IDENTIFIER { $$ = fend_on_identifier($1); FLOC($$, @$); }
	;
	
attribute
	: IDENTIFIER { $$ = fend_on_string_attribute($1); FLOC($$, @$); }
	| TYPE { $$ = fend_on_type_attribute($1); FLOC($$, @$); }
	;

invocation_expression 
	: IDENTIFIER OPEN_PAREN opt_argument_list CLOSE_PAREN { $$ = fend_on_method_call($1, $3); FLOC($$, @$); }
	;

opt_argument_list
	: argument_list { $$ = $1; }
	| { $$ = NULL; }
	;

argument_list
	: expression { $$ = $1; }
	| argument_list COMMA expression { $$ = fend_on_enum($1, $3); FLOC($$, @$); }
	;

typedef
    : type identifier { $$ = fend_on_identifier_declaration($1, $2); FLOC($$, @$); }
	;

type
	: TYPE { $$ = $1; }
	;

%%

void yyerror(char *s, ...)
{
	va_list ap;
	va_start(ap, s);
	char buf[4096];
#ifdef __STDC_WANT_SECURE_LIB__
	vsnprintf_s(buf, sizeof(buf), (size_t)-1, s, ap);
#else
	vsnprintf(buf, sizeof(buf), s, ap);
#endif
	va_end(ap);
	fend_print_error(yylloc.first_line, yylloc.first_column, yylloc.last_line, yylloc.last_column, buf);
	fend_query_cleanup(NULL);
}

void lyyerror(YYLTYPE t, char *s, ...)
{
	va_list ap;
	va_start(ap, s);
	char buf[4096];
	int result;
#ifdef __STDC_WANT_SECURE_LIB__
	result = vsnprintf_s(buf, sizeof(buf), (size_t)-1, s, ap);
#else
	result = vsnprintf(buf, sizeof(buf), s, ap);
#endif
	va_end(ap);
	if (result >= 0)
		fend_print_error(t.first_line, t.first_column, t.last_line, t.last_column, buf);
	else
		fend_print_error(t.first_line, t.first_column, t.last_line, t.last_column, "");
}