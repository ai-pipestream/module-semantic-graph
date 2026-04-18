package ai.pipestream.module.semanticgraph;

import org.junit.jupiter.api.Tag;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a test class or method as a stress test.
 * <p>
 * Stress tests are excluded from the default {@code quarkusIntTest} task.
 * Run them explicitly with:
 * <pre>
 *   ./gradlew quarkusIntTest -Dinclude.stress=true
 * </pre>
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Tag("stress")
public @interface Stress {
}
