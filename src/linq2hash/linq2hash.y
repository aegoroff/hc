%{
	#include "linq2hash.tab.h"

    extern char *yytext;
	extern int fend_error_count;
	void yyerror(char *s, ...);
	void lyyerror(YYLTYPE t, char *s, ...);
	int yylex();
%}

%error-verbose

%code requires
{
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
%token ENDL
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
%type <node> expression
%type <node> unary_expression
%type <node> relational_expr
%type <node> boolean_expression
%type <node> conditional_or_expression
%type <node> conditional_and_expression
%type <node> not_expression
%type <node> exclusive_or_expression
%type <node> anonymous_object
%type <node> anonymous_object_initialization
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
	: from_clause query_body { $$ = fend_query_complete($1, $2); }
	;

query_body
	: opt_query_body_clauses select_or_group_clause query_continuation { $$ = fend_on_query_body($1, $2, $3); }
	;

opt_query_body_clauses
	: { $$ = NULL; }
	| query_body_clauses { $$ = $1; }
	;

query_continuation
	: INTO identifier { fend_register_identifier($2, type_def_user); } query_body { $$ = fend_on_continuation($2, $4); }
	| { $$ = NULL; } %prec LOWER_THAN_INTO 
	;

query_body_clauses
	: query_body_clause { $$ = $1; }
	| query_body_clauses query_body_clause { $$ = fend_on_enum($1, $2); }
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
	: FROM typedef WITHIN expression { $$ = fend_on_from($2, $4); }
	;

let_clause
	: LET identifier ASSIGN expression { $$ = fend_on_let($2, $4); }
	;

where_clause
	: WHERE boolean_expression { $$ = fend_on_where($2); }
	;

join_clause
	: JOIN typedef WITHIN expression ON expression EQUALS expression { $$ = fend_on_join($2, $4, $6, $8); }
	;

join_into_clause
	: JOIN typedef WITHIN expression ON expression EQUALS expression INTO identifier { fend_register_identifier($10, type_def_user); $$ = fend_on_continuation($10, fend_on_join($2, $4, $6, $8)); }
	;

orderby_clause
	: ORDERBY orderings { $$ = fend_on_order_by($2); }
	;

orderings
	: ordering { $$ = $1; }
	| orderings COMMA ordering { $$ = fend_on_enum($1, $3); }
	;

ordering
	: expression ordering_direction { $$ = fend_on_ordering($1, $2); }
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
	: SELECT expression { $$ = $2; }
	;

group_clause
	: GRP expression BY expression { $$ = fend_on_group($2, $4); }
	;

boolean_expression
	: conditional_or_expression
	;

conditional_or_expression
	: conditional_and_expression { $$ = $1; }
	| conditional_or_expression OR conditional_and_expression { $$ = fend_on_predicate($1, $3, node_type_or_rel); }
	;

conditional_and_expression
	: not_expression { $$ = $1; }
	| conditional_and_expression AND not_expression { $$ = fend_on_predicate($1, $3, node_type_and_rel); }
	;

not_expression
	: exclusive_or_expression { $$ = $1; }
	| NOT exclusive_or_expression { $$ = fend_on_predicate($2, NULL, node_type_not_rel); }
	;

exclusive_or_expression
	: OPEN_PAREN boolean_expression CLOSE_PAREN { $$ = $2; }
	| relational_expr { $$ = $1; }
	;

expression
	: unary_expression
	| anonymous_object
	| query_expression
	;
	
unary_expression
	: identifier { if (!fend_is_identifier_defined($1)) lyyerror(@1,"identifier %s undefined", $1->value.string); $$ = fend_on_unary_expression(unary_exp_type_identifier, $1, NULL); }
	| identifier DOT attribute { if (!fend_is_identifier_defined($1)) lyyerror(@1,"identifier %s undefined", $1->value.string); $$ = fend_on_unary_expression(unary_exp_type_property_call, $1, $3); }
	| identifier DOT invocation_expression { if (!fend_is_identifier_defined($1)) lyyerror(@1,"identifier %s undefined", $1->value.string); $$ = fend_on_unary_expression(unary_exp_type_mehtod_call, $1, $3); }
	| STRING { $$ = fend_on_unary_expression(unary_exp_type_string, $1, NULL); }
	| INTEGER { $$ = fend_on_unary_expression(unary_exp_type_number, $1, NULL); }
	;
	
anonymous_object
	: OPEN_BRACE anonymous_object_initialization CLOSE_BRACE { $$ = $2; }
	;
	
anonymous_object_initialization
	: unary_expression { $$ = $1; }
	| anonymous_object_initialization COMMA unary_expression { $$ = fend_on_enum($1, $3); }
	;

relational_expr
	: unary_expression REL_OP unary_expression { $$ = fend_on_releational_expr($1, $3, $2); } // TODO: implement type checking
	;

identifier 
	: IDENTIFIER { $$ = fend_on_identifier($1); }
	;
	
attribute
	: IDENTIFIER { $$ = fend_on_string_attribute($1); }
	| TYPE { $$ = fend_on_type_attribute($1); }
	;

invocation_expression 
	: IDENTIFIER OPEN_PAREN opt_argument_list CLOSE_PAREN {$$ = fend_on_method_call($1, $3); }
	;

opt_argument_list
	: argument_list { $$ = $1; }
	| { $$ = NULL; }
	;

argument_list
	: expression { $$ = $1; }
	| argument_list COMMA expression { $$ = fend_on_enum($1, $3); }
	;

typedef
    : type identifier { $$ = fend_on_identifier_declaration($1, $2); }
	| identifier { $$ = fend_on_identifier_declaration(fend_on_simple_type_def(type_def_dynamic), $1); }
	;

type
	: TYPE { $$ = $1; }
	| IDENTIFIER { $$ = fend_on_complex_type_def(type_def_user, $1); }
	;

%%

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