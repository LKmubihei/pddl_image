(define (problem picture_176)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator blue_regulator green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_regulator)
        (clear blue_regulator)
        (clear green_regulator)
        (part_at red_regulator table)
        (part_at blue_regulator table)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
        )
    )
)
