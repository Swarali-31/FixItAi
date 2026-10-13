# Diagnosing Java NullPointerException
Category: Java runtime errors
Technologies: Java

A NullPointerException means code attempted to use a null reference. Read the complete stack trace and start at the first frame belonging to your application. Identify the exact expression and determine which operand is null. Check method return values, object initialization, dependency injection, deserialization, and collection lookups. Add a focused assertion or debugger breakpoint immediately before the failing line.

Fix the source of the missing value rather than catching and ignoring the exception. Validate external inputs at boundaries and use explicit error handling when a value is legitimately optional. In Java 14 and newer, helpful NullPointerException messages can identify which expression was null. Include the Java version and the relevant stack-trace frames when searching for a framework-specific cause.
